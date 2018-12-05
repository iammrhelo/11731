#!/bin/sh

vocab="debug/vocab.en-de.bin"
train_src="debug/train.en-de.en"
train_tgt="debug/train.en-de.de"
dev_src="debug/valid.en-de.en"
dev_tgt="debug/valid.en-de.de"

work_dir="work_dir.debug"

mkdir -p ${work_dir}
echo save results to ${work_dir}

batch_size=2
a=$(wc -l < "${train_src}")
b=$batch_size
valid_niter=$((a%b?a/b+1:a/b))

python gnmt_skeleton.py \
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

