#!/bin/bash

cd ../..

# custom config
DATA=/home/harim/data/cocoop_dataset
TRAINER=CoCoOp_Prototype
TRAIN=CoCoOp

DATASET=$1
SEED=$2
LOADEP=$3

# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
SHOTS=16
SUB=base

CFG=vit_b16_c4_ep${LOADEP}_batch1_ctxv1_prototype

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/base2new/train_base_e${LOADEP}/${COMMON_DIR}
# MODEL_DIR=output/base2base/train_base/${COMMON_DIR}
# MODEL_DIR=output/base2base/train_base/imagenet/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

DIR=output/base2base/test_${SUB}_e${LOADEP}/${COMMON_DIR}
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
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi