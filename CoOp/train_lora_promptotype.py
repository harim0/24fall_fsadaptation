import argparse
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
import os
import json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.data.datasets import build_dataset
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.lora_promptotype
import trainers.zsclip


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.LORA_PROMPTOTYPE = CN()
    cfg.TRAINER.LORA_PROMPTOTYPE.N_CTX = 16  # number of context vectors
    cfg.TRAINER.LORA_PROMPTOTYPE.CTX_INIT = "a photo of"  # initialization words
    cfg.TRAINER.LORA_PROMPTOTYPE.CTX_MID = ", which is a"
    cfg.TRAINER.LORA_PROMPTOTYPE.PREC = "amp"  # fp16, fp32, amp
    
    
    cfg.TRAINER.LORA_PROMPTOTYPE.POSITION="all"
    cfg.TRAINER.LORA_PROMPTOTYPE.ENCODER="both"
    cfg.TRAINER.LORA_PROMPTOTYPE.PARAMS=["q","k","v"]
    cfg.TRAINER.LORA_PROMPTOTYPE.RANK=2
    cfg.TRAINER.LORA_PROMPTOTYPE.ALPHA=1
    cfg.TRAINER.LORA_PROMPTOTYPE.DROPOUT_RATE=0.25
    
    cfg.OPTIM_LORA = CN()
    cfg.OPTIM_LORA.NAME = "adamw"
    cfg.OPTIM_LORA.LR = 2e-4
    cfg.OPTIM_LORA.MAX_EPOCH = 10
    cfg.OPTIM_LORA.LR_SCHEDULER = "cosine"
    cfg.OPTIM_LORA.WARMUP_EPOCH = 1
    cfg.OPTIM_LORA.WARMUP_TYPE = "constant"
    cfg.OPTIM_LORA.WARMUP_CONS_LR = 1e-5
    cfg.OPTIM_LORA.WARMUP_MIN_LR = 1e-6
    cfg.OPTIM_LORA.WARMUP_RECOUNT = False
    cfg.OPTIM_LORA.WEIGHT_DECAY = 0.01
    cfg.OPTIM_LORA.MOMENTUM = 0.9
    cfg.OPTIM_LORA.SGD_DAMPNING = 0  # 모멘텀 감쇠율
    cfg.OPTIM_LORA.SGD_NESTEROV = False  # Nesterov 모멘텀 사용 여부
    cfg.OPTIM_LORA.RMSPROP_ALPHA = 0.99  # RMSProp 감쇠율
    cfg.OPTIM_LORA.STAGED_LR = False  # 단계적 학습률 조정 여부
    cfg.OPTIM_LORA.NEW_LAYERS = []  # 새로운 레이어 (예: [])
    cfg.OPTIM_LORA.BASE_LR_MULT = 1.0  # 기본 학습률 배수
    cfg.OPTIM_LORA.ADAM_BETA1 = 0.9  # Adam 1차 모멘텀 베타 값
    cfg.OPTIM_LORA.ADAM_BETA2 = 0.999  # Adam 2차 모멘텀 베타 값
    cfg.OPTIM_LORA.STEPSIZE = 5  # StepLR에 필요한 단계 크기
    cfg.OPTIM_LORA.GAMMA = 0.1 

    cfg.OPTIM_PROMPTLEARNER = CN()
    cfg.OPTIM_PROMPTLEARNER.NAME = "sgd"
    cfg.OPTIM_PROMPTLEARNER.LR = 0.002
    cfg.OPTIM_PROMPTLEARNER.MAX_EPOCH = 10
    cfg.OPTIM_PROMPTLEARNER.LR_SCHEDULER = "cosine"
    cfg.OPTIM_PROMPTLEARNER.WARMUP_EPOCH = 1
    cfg.OPTIM_PROMPTLEARNER.WARMUP_TYPE = "constant"
    cfg.OPTIM_PROMPTLEARNER.WARMUP_CONS_LR = 1e-5
    cfg.OPTIM_PROMPTLEARNER.WARMUP_MIN_LR = 1e-6
    cfg.OPTIM_PROMPTLEARNER.WARMUP_RECOUNT = True
    cfg.OPTIM_PROMPTLEARNER.WEIGHT_DECAY = 0.01
    cfg.OPTIM_PROMPTLEARNER.MOMENTUM = 0.9
    cfg.OPTIM_PROMPTLEARNER.SGD_DAMPNING = 0  # 모멘텀 감쇠율
    cfg.OPTIM_PROMPTLEARNER.SGD_NESTEROV = False  # Nesterov 모멘텀 사용 여부
    cfg.OPTIM_PROMPTLEARNER.RMSPROP_ALPHA = 0.99  # RMSProp 감쇠율
    cfg.OPTIM_PROMPTLEARNER.STAGED_LR = False  # 단계적 학습률 조정 여부
    cfg.OPTIM_PROMPTLEARNER.NEW_LAYERS = []  # 새로운 레이어 (예: [])
    cfg.OPTIM_PROMPTLEARNER.BASE_LR_MULT = 1.0  # 기본 학습률 배수
    cfg.OPTIM_PROMPTLEARNER.ADAM_BETA1 = 0.9  # Adam 1차 모멘텀 베타 값
    cfg.OPTIM_PROMPTLEARNER.ADAM_BETA2 = 0.999  # Adam 2차 모멘텀 베타 값
    cfg.OPTIM_PROMPTLEARNER.STEPSIZE = 10  # Adam 2차 모멘텀 베타 값
    cfg.OPTIM_PROMPTLEARNER.GAMMA = 0.1  # Adam 2차 모멘텀 베타 값

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    
def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


# def main(args):
def main_worker(rank, world_size, args):
    cfg = setup_cfg(args)
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15280"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  
    
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))
    
    trainer = build_trainer(cfg, rank=rank, world_size=world_size)
    dataset = build_dataset(cfg)
    trainer.superclass_mapping = dataset.SUPERCLASS_MAPPING

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()
        
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/harim/data/cocoop_dataset", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="/home/harim/ai_finalproject/CoOp/output", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    
    
    device_ids = list(range(torch.cuda.device_count()))  
    world_size = torch.cuda.device_count()

    spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    