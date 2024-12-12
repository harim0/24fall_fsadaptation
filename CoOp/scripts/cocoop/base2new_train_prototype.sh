#!/bin/bash

cd ../..

# custom config
DATA=/home/harim/data/cocoop_dataset
TRAINER=CoCoOp_Prototype
TRAIN=CoCoOp
# TRAINER=CoOp

DATASET=$1
SEED=$2
EPOCH=$3

CFG=vit_b16_c4_ep${EPOCH}_batch1_ctxv1_prototype
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=16


DIR=output/base2new/train_base_e${EPOCH}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
# DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train_prototype.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAIN}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi