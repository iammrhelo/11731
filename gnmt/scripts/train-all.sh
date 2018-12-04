#!/bin/sh

vocab="data/vocab.all.bin"
train_src="data/train.all.src"
train_tgt="data/train.all.tgt"
dev_src="data/valid.all.src"
dev_tgt="data/valid.all.tgt"

work_dir="work_dir-all"

mkdir -p ${work_dir}
echo save results to ${work_dir}

batch_size=64
a=$(wc -l < "${train_src}")
b=$batch_size
valid_niter=$((a%b?a/b+1:a/b))

python -u gnmt_skeleton.py \
    train \
    --cuda \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir} \
    --num-layers 2 \
    --max-epoch 30 \
    --valid-niter ${valid_niter} \
    --batch-size ${batch_size} \
    --hidden-size 256 \
    --embed-size 256 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 \
    --bidirectional  \

#>${work_dir}/err.log

