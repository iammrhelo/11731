#!/bin/sh

vocab="data.debug/vocab.bin"
train_src="data.debug/train.de-en.de.wmixerprep"
train_tgt="data.debug/train.de-en.en.wmixerprep"
dev_src="data.debug/valid.de-en.de"
dev_tgt="data.debug/valid.de-en.en"
test_src="data.debug/test.de-en.de"
test_tgt="data.debug/test.de-en.en"
#test_src=$train_src
#test_tgt=$train_tgt
work_dir="work_dir.debug"

mkdir -p ${work_dir}
echo save results to ${work_dir}

python nmt.py \
    train \
    --cuda \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir} \
    --bidirectional  \
    --attn-type General \
    --num-layers 1 \
    --valid-niter 1000 \
    --batch-size 2 \
    --hidden-size 20 \
    --embed-size 20 \
    --uniform-init 0.1 \
    --dropout 0.0 \
    --clip-grad 5.0 \
    --lr-decay 0.5 \
    --max-epoch 1000 \
    --patience 1000 \
    #--model-path ${work_dir}/model.bin
#>${work_dir}/err.log

