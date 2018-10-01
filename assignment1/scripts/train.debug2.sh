#!/bin/sh

vocab="data.debug2/vocab.bin"
train_src="data.debug2/train.de-en.de.wmixerprep"
train_tgt="data.debug2/train.de-en.en.wmixerprep"
dev_src="data.debug2/valid.de-en.de"
dev_tgt="data.debug2/valid.de-en.en"
test_src="data.debug2/test.de-en.de"
test_tgt="data.debug2/test.de-en.en"
dev_src=$train_src
dev_tgt=$train_tgt
#test_src=$train_src
#test_tgt=$train_tgt
work_dir="work_dir.debug2"

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
    --num-layers 1 \
    --valid-niter 10 \
    --batch-size 10 \
    --hidden-size 200 \
    --embed-size 100 \
    --uniform-init 0.1 \
    --dropout 0.0 \
    --clip-grad 5.0 \
    --lr-decay 0.5 \
    --max-epoch 100 \
    --patience 5 \
    --model-path ${work_dir}/model.bin
#>${work_dir}/err.log

