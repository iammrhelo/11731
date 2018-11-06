#!/bin/sh

l1=$1
l2=$2

vocab="data/vocab.$l1+$l2-en.bin"
train_src="data/train.en-$l1+$l2.$l1+$l2.txt"
train_tgt="data/train.en-$l1+$l2.en.txt"
dev_src="data/dev.en-$l1+$l2.$l1+$l2.txt"
dev_tgt="data/dev.en-$l1+$l2.en.txt"
test_src="data/dev.en-$l1+$l2.$l1+$l2.txt"
test_tgt="data/dev.en-$l1+$l2.en.txt"

embed_weights="align_embed/data.weights/wiki.$l1+$l2-en.weights"

work_dir="work_dir.$l1+$l2-en.embed"

mkdir -p ${work_dir}
echo save results to ${work_dir}

#Parameters
batch_size=48
a=$(wc -l < "${train_src}")
b=$batch_size
valid_niter=$((a%b?a/b+1:a/b))
echo valid_niter: $valid_niter
python -u embed_nmt.py \
    train \
    --cuda \
    --embed-pretrain \
    --embed-weights ${embed_weights} \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir} \
    --num-layers 1 \
    --attn-type General \
    --max-epoch 40 \
    --valid-niter $valid_niter \
    --batch-size $batch_size \
    --hidden-size 256 \
    --embed-size 256 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 \
    --bidirectional  \

#>${work_dir}/err.log


