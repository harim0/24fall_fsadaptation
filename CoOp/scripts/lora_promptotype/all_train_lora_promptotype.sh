#!/bin/bash

cd ../..

# custom config
DATA=/home/harim/data/cocoop_dataset
TRAINER=LoRA_Promptotype

DATASET=$1
SEED=$2
EPOCH=$3

CFG=vit_b16_c4_ep${EPOCH}_batch1_ctxv1_lora_promptotype
SHOTS=16


DIR=output/all/train_all_e${EPOCH}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train_lora_promptotype.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all 
fi