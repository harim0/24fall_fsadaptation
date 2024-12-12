import os.path as osp
from collections import OrderedDict
import math
import json
import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset

from dassl.utils import read_image
from dassl.data import DataManager
from dassl.data.datasets import build_dataset
from dassl.data.data_manager import DatasetWrapper

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class DatasetWrapperWithSuperclass(Dataset):
    def __init__(self, data_source, transform, superclass_mapping, lab2cname):
        self.data_source = data_source
        self.transform = transform
        self.superclass_mapping = superclass_mapping
        self.lab2cname = lab2cname
        self.superclass2lab = {v: i for i, v in enumerate(set(superclass_mapping.values()))}

    def __getitem__(self, index):
        item = self.data_source[index]
        # print(f"Datum object: {item}")  # 객체 출력
        # print(f"Attributes: {dir(item)}")  # 객체 속성 확인
        # '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_classname', '_domain', '_impath', '_label', 'classname', 'domain', 'impath', 'label'
        img = read_image(item.impath)
        label = item.label
        classname = self.lab2cname[label]
        superclass = self.superclass_mapping[classname]
        superclass_label = self.superclass2lab[superclass]

        return {
            "img": self.transform(img),
            "label": label,
            "superclass_label": superclass_label,
        }

    def __len__(self):
        return len(self.data_source)
    
    def get_unique_classnames(self):
        return {self.lab2cname[item.label] for item in self.data_source}

    def get_superclass_keys(self):
        return set(self.superclass_mapping.keys())

    def compare_classnames_and_superclass_keys(self):
        classnames = self.get_unique_classnames()
        superclass_keys = self.get_superclass_keys()

        missing_in_superclass = classnames - superclass_keys
        missing_in_classnames = superclass_keys - classnames

        return {
            "missing_in_superclass": missing_in_superclass,
            "missing_in_classnames": missing_in_classnames,
        }

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, superclass_mapping, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP_PROMPTOTYPE.N_CTX
        ctx_init = cfg.TRAINER.COCOOP_PROMPTOTYPE.CTX_INIT
        ctx_mid = cfg.TRAINER.COCOOP_PROMPTOTYPE.CTX_MID
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        
        
        self.superclasses = list(set(superclass_mapping.values()))  # Superclass 리스트 생성
        self.superclass_to_id = {sc: i for i, sc in enumerate(self.superclasses)}  # Superclass ID 매핑
        self.superclass_embeddings = nn.Embedding(len(self.superclasses), ctx_dim)  # Superclass-specific embedding

        
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx1_vectors = embedding[0, 1 : 1 + n_ctx, :]
        else:
            ctx1_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx1_vectors, std=0.02)

        self.ctx1 = nn.Parameter(ctx1_vectors)
        
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        if cfg.TRAINER.COCOOP_PROMPTOTYPE.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        superclass_mapping = {k.replace("_", " "): v.replace("_", " ") for k, v in superclass_mapping.items()}
        
        prompts = [
            f"{ctx_init} {name} {ctx_mid} {superclass_mapping[name]}."
            for name in classnames
        ]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # tokenized_prompts = tokenized_prompts[:, :clip_model.positional_embedding.size(0)]
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)


        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        prefix_len = 1
        midfix_len = len(_tokenizer.encode(ctx_mid))
        suffix_start = prefix_len + n_ctx + midfix_len

        self.register_buffer("token_prefix", embedding[:, :prefix_len, :])  # Prefix (e.g., "A photo of a")
        self.register_buffer("token_midfix", embedding[:, prefix_len + n_ctx : suffix_start, :])  # Midfix
        self.register_buffer("token_suffix", embedding[:, suffix_start:, :])  # Suffix (e.g., ".", CLS, EOS)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        
    def get_superclass_id(self, class_name):
        return self.superclass_to_id[self.superclass_mapping[class_name]]
    
    def construct_prompts(self, ctx1, prefix, midfix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            midfix = midfix[label]
            suffix = suffix[label]
        
            # Superclass-specific embedding 추가
            superclass_ids = torch.tensor([self.get_superclass_id(self.classnames[l]) for l in label])
            superclass_embeds = self.superclass_embeddings(superclass_ids).unsqueeze(1)  # (batch, 1, ctx_dim)

            midfix = midfix + superclass_embeds

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx1,     # (dim0, n_ctx, dim)
                midfix,
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        midfix = self.token_midfix
        suffix = self.token_suffix
        ctx1 = self.ctx1                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx1 = ctx1.unsqueeze(0)             # (1, n_ctx1, ctx1_dim)
        ctx1_shifted = ctx1 + bias           # (batch, n_ctx, ctx_dim)
        prompts = []
        for ctx_shifted_i in ctx1_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, midfix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, superclass_names, clip_model):
        super().__init__()
        self.prompt_learner =PromptLearner(cfg, classnames, superclass_names, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None, superclass_label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        
        logits_class = []
        logits_superclass = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_class.append(logit_scale * imf_i @ text_features.t())

            superclass_logits = logit_scale * imf_i.float() @ self.prompt_learner.superclass_embeddings.weight.t()
            logits_superclass.append(superclass_logits)
            
        logits_class = torch.stack(logits_class)
        logits_superclass = torch.stack(logits_superclass)
        
        if self.prompt_learner.training:
            loss_class = F.cross_entropy(logits_class, label) if label is not None else 0
            loss_superclass = F.cross_entropy(logits_superclass, superclass_label)
            alpha = 0.5  
            loss = loss_class + alpha * loss_superclass
            return image_features, loss

        return logits_class


@TRAINER_REGISTRY.register()
class CoCoOp_Promptotype(TrainerX):
    def __init__(self, cfg, rank, world_size):
        super().__init__(cfg, rank, world_size)
        assert self.rank is not None, "\t\tRank is not set properly in CoCoOp_Prototype"
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP_PROMPTOTYPE.PREC in ["fp16", "fp32", "amp"]
        
    def build_data_loader(self):
        cfg = self.cfg
        dataset = build_dataset(cfg)
        self.superclass_mapping = dataset.SUPERCLASS_MAPPING
        # print("\t\tDataset SUPERCLASS_MAPPING:", dataset.SUPERCLASS_MAPPING)
        # print("\t\tSample Data:", dataset.train_x[0]) 
        
        print(f"\t\tDataset classes: {dataset.num_classes}")

        assert len(dataset.train_x) > 0, "Error: train_x is empty! Check data loading."
        dm = DataManager(self.cfg)
        self.dm = dm

        train_transform = T.Compose([
            T.RandomResizedCrop(cfg['INPUT']['SIZE']),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=cfg['INPUT']['PIXEL_MEAN'],
                std=cfg['INPUT']['PIXEL_STD']
            ),
        ])

        test_transform = T.Compose([
            T.Resize(cfg['INPUT']['SIZE']),
            T.ToTensor(),
            T.Normalize(
                mean=cfg['INPUT']['PIXEL_MEAN'],
                std=cfg['INPUT']['PIXEL_STD']
            ),
        ])

        train_dataset = DatasetWrapperWithSuperclass(
            data_source=dataset.train_x,
            transform=train_transform,
            superclass_mapping=dataset.SUPERCLASS_MAPPING,
            lab2cname=dataset.lab2cname,
        )
        test_dataset = DatasetWrapperWithSuperclass(
            data_source=dataset.test,
            transform=test_transform,
            superclass_mapping=dataset.SUPERCLASS_MAPPING,
            lab2cname=dataset.lab2cname,
        )

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )

        self.train_loader_x = DataLoader(
            dataset=train_dataset,
            batch_size=cfg['DATALOADER']['TRAIN_X']['BATCH_SIZE'],
            sampler=train_sampler,
            num_workers=cfg['DATALOADER']['NUM_WORKERS'],
            pin_memory=True,
            drop_last=True
        )

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=cfg['DATALOADER']['TEST']['BATCH_SIZE'],
            sampler=test_sampler,
            num_workers=cfg['DATALOADER']['NUM_WORKERS'],
            pin_memory=True,
            drop_last=False
        )
        self.num_classes = dataset.num_classes
        self.num_source_domains = len(cfg['DATASET']['SOURCE_DOMAINS'])
        self.lab2cname = dataset.lab2cname
        
    def build_visual_protocriticism(self, cfg, clip_model, train_loader):
        model_dir_root = '/home/harim/ai_finalproject/CoOp/cache'
        os.makedirs(model_dir_root, exist_ok=True)

        key_path = f"{model_dir_root}/visual_key.pt"
        value_path = f"{model_dir_root}/visual_value.pt"

        if os.path.exists(key_path) and os.path.exists(value_path):
            cache_keys = torch.load(key_path, weights_only=True)
            cache_values = torch.load(value_path, weights_only=True)
        else:
            cache_keys = []
            cache_values = []
            
            # clip_model = clip_model.to("cpu") 
            clip_model.to(self.device)

            with torch.no_grad():
                for self.batch_idx, batch in enumerate(tqdm(train_loader)):
                    images, labels = self.parse_batch_train(batch)
                    images = images.to(self.device).type(clip_model.dtype)
                    # images = images.to("cpu") 
                    # labels = labels.to("cpu")
                    
                    image_features = clip_model.encode_image(images) # (batch_size, feature_dim)
                    cache_keys.append(image_features) 
                    cache_values.append(labels)

            cache_keys = torch.cat(cache_keys, dim=0) # (num_samples, feature_dim) torch.Size([1000, 512]), torch.float16
            cache_values = torch.cat(cache_values, dim=0) # torch.Size([1000]), range: min=0, max=499, torch.int64
            
            # sorting
            index = torch.argsort(cache_values) # 0-999, torch.Size([1000])
            cache_values = cache_values[index]
            cache_keys = cache_keys[index]
            cache_values = F.one_hot(cache_values) # 정수형 텐서 입력 필요, float 타입일 경우 오류

            torch.save(cache_keys, key_path)
            torch.save(cache_values, value_path)
            
        return cache_keys, cache_values
    
    def calculate_prototypes(self, visual_keys, visual_values, num_classes):
        prototypes = torch.zeros((num_classes, visual_keys.size(1)), device=visual_keys.device)
        counts = torch.zeros(num_classes, device=visual_keys.device)

        for i in range(num_classes):
            class_mask = visual_values[:, i].bool() 
            class_features = visual_keys[class_mask]  

            if class_features.size(0) > 0:
                prototypes[i] = class_features.mean(dim=0)
                counts[i] = class_features.size(0)
            else:
                prototypes[i] = torch.zeros_like(prototypes[i]) 
                
        return prototypes, counts
    
    
    def compute_prototype_loss(self, image_features, labels, alpha=0.7):
        
        assert torch.all(torch.isfinite(image_features)), "image_features contains NaN or Inf"
        assert torch.all(torch.isfinite(self.visual_prototypes)), "visual_prototypes contains NaN or Inf"
        
        proto_norm = self.visual_prototypes.norm(dim=-1, keepdim=True)
        proto_norm = torch.where(proto_norm == 0, torch.tensor(1.0, device=proto_norm.device), proto_norm)
        prototypes = self.visual_prototypes / proto_norm
        
        image_features = image_features.float()
        feature_norm = image_features.norm(dim=-1, keepdim=True)
        feature_norm = torch.where(feature_norm == 0, torch.tensor(1.0, device=feature_norm.device), feature_norm)
        image_features = image_features / feature_norm
        
        assert torch.all(torch.isfinite(prototypes)), "Normalized prototypes contain NaN or Inf"
        assert torch.all(torch.isfinite(image_features)), "Normalized image_features contain NaN or Inf"

        assert image_features.device == self.device
        assert prototypes.device == self.device
        
        similarity = torch.matmul(image_features, prototypes.t()) # 1 x 500
        assert torch.all(torch.isfinite(similarity)), "similarity contains NaN or Inf"
        
        labels = labels.to(torch.long) # torch.float32 -> torch.long 
        # print("labels.shape:", labels.shape) # eurosat 5,512
        if labels.ndim > 1 and labels.size(-1) > 1:  # One-hot encoded 상태
            labels = torch.argmax(labels, dim=0)  # eurosat 숫자 라벨로 변환 torch.Size([5])
            

        # print("similarity.shape:", similarity.shape) # eurosat 1,5
        target_similarity = similarity[torch.arange(similarity.size(0)), labels] 
        target_loss = - target_similarity.mean() # 같은 클래스 프로토타입과의 관계
        
        class_losses = [] # Class-Wise 방식으로 세부적인 클래스 간 관계를 학습
        for class_idx in range(similarity.size(1)): 
            if (labels == class_idx).any():
                continue
            class_loss = similarity[:, class_idx].mean()  
            class_losses.append(class_loss)

        negative_loss = torch.stack(class_losses).mean()
        
        prototype_loss = alpha * (1 + target_loss) + (1 - alpha) * negative_loss
        return prototype_loss
        
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP_PROMPTOTYPE.PREC == "fp32" or cfg.TRAINER.COCOOP_PROMPTOTYPE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            
        visual_memory_keys, visual_memory_values = self.build_visual_protocriticism(cfg, clip_model, self.train_loader_x)
        visual_prototypes, visual_cnts = self.calculate_prototypes(visual_memory_keys, visual_memory_values, self.dm.num_classes)
        
        self.visual_prototypes = nn.Parameter(
            visual_prototypes.clone(), requires_grad=False
        )
        self.visual_prototypes = self.visual_prototypes.to(self.device)
        
        clip_model.to("cpu")

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, self.superclass_mapping, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP_PROMPTOTYPE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            # self.model = nn.DataParallel(self.model)
            self.model = DDP(self.model, device_ids=[self.device], output_device=self.rank)

    def forward_backward(self, batch):
        image, label, superclass_label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        features, loss = None, None
        
        prec = self.cfg.TRAINER.COCOOP_PROMPTOTYPE.PREC
        if prec == "amp":
            with autocast():
                features, task_loss = model(image, label, superclass_label)
        else:
            features, task_loss = model(image, label, superclass_label)

        
        features = features.to(self.device)
        proto_loss = self.compute_prototype_loss(features, self.visual_prototypes)
        alpha = 0.6
        loss = alpha * task_loss + (1-alpha) * proto_loss
        
        if features is None or loss is None:
            raise ValueError("Model forward pass did not return features or task_loss.")

        optim.zero_grad()
        if prec == "amp":
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()
        
        loss_summary = {"task_loss": task_loss.item(), "proto_loss": proto_loss.item(), "loss": loss.item()}


        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        superclass_label = batch["superclass_label"]
        input = input.to(self.device)
        label = label.to(self.device)
        superclass_label = superclass_label.to(self.device)
        return input, label, superclass_label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]
                
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
