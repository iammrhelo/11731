#!/bin/sh

name='nl-en-de'
vocab="../iwslt2017/data/vocab2.${name}.bin"
train_src="../iwslt2017/data/train2.${name}.src"
train_tgt="../iwslt2017/data/train2.${name}.tgt"
dev_src="../iwslt2017/data/valid2.${name}.src"
dev_tgt="../iwslt2017/data/valid2.${name}.tgt"

work_dir="work_dir2-${name}"

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
    --num-layers 1 \
    --max-epoch 30 \
    --log-every 1000 \
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

