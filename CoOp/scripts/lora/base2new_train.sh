#!/bin/bash

# custom config
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
DATA=../datasets/
TRAINER=LoRA
# TRAINER=CoOp

DATASET=$1
SEED=$2

CFG=vit_b16_ep10_batch32
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=16


DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi