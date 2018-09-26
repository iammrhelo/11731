#!/bin/sh

vocab="data.debug/vocab.bin"
train_src="data.debug/train.de-en.de.wmixerprep.debug"
train_tgt="data.debug/train.de-en.en.wmixerprep.debug"
dev_src="data.debug/valid.de-en.de.debug"
dev_tgt="data.debug/valid.de-en.en.debug"
test_src="data.debug/test.de-en.de.debug"
test_tgt="data.debug/test.de-en.en.debug"

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
    --valid-niter 10 \
    --batch-size 8 \
    --hidden-size 50 \
    --embed-size 50 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 

#>${work_dir}/err.log

