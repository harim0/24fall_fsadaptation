import os.path as osp
from collections import OrderedDict
import math
import json

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
        n_ctx = cfg.TRAINER.COCOOP_PROMPT.N_CTX
        ctx_init = cfg.TRAINER.COCOOP_PROMPT.CTX_INIT
        ctx_mid = cfg.TRAINER.COCOOP_PROMPT.CTX_MID
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        
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
        # ctx2_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        # nn.init.normal_(ctx2_vectors, std=0.02)
        # self.ctx2 = nn.Parameter(ctx2_vectors)
        
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        if cfg.TRAINER.COCOOP_PROMPT.PREC == "fp16":
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
    
    # def construct_prompts(self, ctx1, ctx2, prefix, midfix, suffix, label=None):
    def construct_prompts(self, ctx1, prefix, midfix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            midfix = midfix[label]
            suffix = suffix[label]
        
        # print(f"prefix size: {prefix.size()}") #  torch.Size([9, 1, 512])
        # print(f"ctx1 size: {ctx1.size()}") # torch.Size([50, 4, 512]
        # print(f"midfix size: {midfix.size()}") #  torch.Size([9, 4, 512])
        # print(f"ctx2 size: {ctx2.size()}") # torch.Size([50, 4, 512])
        # print(f"suffix size: {suffix.size()}") # torch.Size([9, 68, 512])
        # print(f"label: {label}")
        # print(f"n_cls: ",self.n_cls)
        # print(f"n_ctx: ",self.n_ctx)
        # print(f"Tokenized prompts shape: {self.tokenized_prompts.size()}")

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx1,     # (dim0, n_ctx, dim)
                midfix,
                # ctx2,
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
        # ctx2 = self.ctx2                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx1 = ctx1.unsqueeze(0)             # (1, n_ctx1, ctx1_dim)
        # ctx2 = ctx2.unsqueeze(0)             # (1, n_ctx2, ctx2_dim)
        ctx1_shifted = ctx1 + bias           # (batch, n_ctx, ctx_dim)
        # ctx2_shifted = ctx2 + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        # for ctx1_shifted_i, ctx2_shifted_i in zip(ctx1_shifted, ctx2_shifted):
        #     ctx1_i = ctx1_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
        #     ctx2_i = ctx2_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
        #     pts_i = self.construct_prompts(ctx1_i, ctx2_i, prefix, midfix, suffix)
        #     prompts.append(pts_i)
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
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        
        if self.prompt_learner.training:
            loss_class = F.cross_entropy(logits, label) if label is not None else 0
            loss_superclass = F.cross_entropy(logits, superclass_label)
            alpha = 0.5
            loss = loss_class + loss_superclass*alpha
            return loss

        return logits


@TRAINER_REGISTRY.register()
class CoCoOp_Prompt(TrainerX):
    def __init__(self, cfg, rank, world_size):
        super().__init__(cfg, rank, world_size)
        assert self.rank is not None, "\t\tRank is not set properly in CoCoOp_Prototype"
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP_PROMPT.PREC in ["fp16", "fp32", "amp"]
        
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
        
        # print()
        # print()
        # comparison_result = train_dataset.compare_classnames_and_superclass_keys()
        # print("\t\tClassnames but not in superclass keys:", comparison_result["missing_in_superclass"])
        # print("len = ",len(comparison_result["missing_in_superclass"]))
        print()
        comparison_result = test_dataset.compare_classnames_and_superclass_keys()
        print("\t\tClassnames but not in superclass keys:", comparison_result["missing_in_superclass"])
        print("len = ",len(comparison_result["missing_in_superclass"]))
        print("\t\tSuperclass keys but not in classnames:", comparison_result["missing_in_classnames"])
        print("len= ",len(comparison_result["missing_in_classnames"]))
        print()
        print()

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
        
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP_PROMPT.PREC == "fp32" or cfg.TRAINER.COCOOP_PROMPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

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

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP_PROMPT.PREC == "amp" else None

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
        
        prec = self.cfg.TRAINER.COCOOP_PROMPT.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label, superclass_label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label, superclass_label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

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
