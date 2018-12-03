#!/bin/sh

vocab="data/vocab.all.bin"
train_src="data/train.all.src"
train_tgt="data/train.all.tgt"
dev_src="data/valid.all.src"
dev_tgt="data/valid.all.src"
test_src="data/test.all.src"
test_tgt="data/test.all.src"

work_dir="work_dir-all"

mkdir -p ${work_dir}
echo save results to ${work_dir}

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
    --max-epoch 20 \
    --valid-niter 600 \
    --batch-size 64 \
    --hidden-size 256 \
    --embed-size 256 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 \
    --bidirectional  \

#>${work_dir}/err.log

