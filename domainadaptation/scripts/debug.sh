#!/bin/sh

vocab="data/vocab.debug.bin"
train_src="data/debug.train.de"
train_tgt="data/debug.train.en"
dev_src="data/debug.valid.de"
dev_tgt="data/debug.valid.en"

work_dir="work_dir.debug"

mkdir -p ${work_dir}
echo save results to ${work_dir}

batch_size=10
a=$(wc -l < "${train_src}")
b=$batch_size
valid_niter=$((a%b?a/b+1:a/b))

python hypernmt.py \
    train \
    --cuda \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir} \
    --num-layers 1 \
    --max-epoch 10 \
    --valid-niter ${valid_niter} \
    --batch-size ${batch_size} \
    --hidden-size 50 \
    --embed-size 50 \
    --uniform-init 0.1 \
    --dropout 0.0 \
    --clip-grad 5.0 \
    --lr-decay 0.5 \
    --bidirectional  \

#>${work_dir}/err.log

