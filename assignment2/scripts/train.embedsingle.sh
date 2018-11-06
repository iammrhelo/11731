#!/bin/sh

l1=$1

vocab="data/vocab.$l1-en.bin"
train_src="data/train.en-$l1.$l1.txt"
train_tgt="data/train.en-$l1.en.txt"
dev_src="data/dev.en-$l1.$l1.txt"
dev_tgt="data/dev.en-$l1.en.txt"
test_src="data/dev.en-$l1.$l1.txt"
test_tgt="data/dev.en-$l1.en.txt"

embed_src="align_embed/data.weights/wiki.$l1-en.weights"
embed_tgt="align_embed/data.weights/wiki.en-en.weights"

work_dir="work_dir.$l1-en.embed"

mkdir -p ${work_dir}
echo save results to ${work_dir}


#Parameters
batch_size=64
a=$(wc -l < "${train_src}")
b=$batch_size
valid_niter=$((a%b?a/b+1:a/b))
echo valid_niter: $valid_niter
python -u embed_nmt.py \
    train \
    --cuda \
    --embed-pretrain \
    --embed-src ${embed_src} \
    --embed-tgt ${embed_tgt} \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir} \
    --num-layers 1 \
    --attn-type General \
    --max-epoch 30 \
    --valid-niter $valid_niter \
    --batch-size $batch_size \
    --hidden-size 512 \
    --embed-size 300 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 \
    --bidirectional  \

#>${work_dir}/err.log


