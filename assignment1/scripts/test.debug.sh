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
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt
