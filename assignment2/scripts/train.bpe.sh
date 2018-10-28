#!/bin/sh

l1=$1
l2=$2

vocab="data.bpe/vocab.$l1+$l2-en.bpe.bin"
train_src="data.bpe/train.en-$l1+$l2.$l1+$l2.bpe.txt"
train_tgt="data.bpe/train.en-$l1+$l2.en.bpe.txt"
dev_src="data.bpe/dev.en-$l1.$l1.bpe.txt"
dev_tgt="data.bpe/dev.en-$l1.en.bpe.txt"
test_src="data.bpe/dev.en-$l1.$l1.bpe.txt"
test_tgt="data.bpe/dev.en-$l1.en.bpe.txt"

work_dir="work_dir.$l1+$l2-en.bpe"

mkdir -p ${work_dir}
echo save results to ${work_dir}


#Parameters
batch_size=16
a=$(wc -l < "${train_src}")
b=$batch_size
valid_niter=$(((a%b?a/b+1:a/b)/2))
echo valid_niter: $valid_niter
python nmt.py \
    train \
    --cuda \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir} \
    --num-layers 1 \
    --attn-type General \
    --max-epoch 15 \
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

