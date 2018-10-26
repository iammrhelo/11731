#!/bin/sh

l1=$1
beam_size=$2

vocab="data/vocab.$l1-en.bin"
train_src="data/train.en-$l1.$l1.txt"
train_tgt="data/train.en-$l1.en.txt"
dev_src="data/dev.en-$l1.$l1.txt"
dev_tgt="data/dev.en-$l1.en.txt"
test_src="data/test.en-$l1.$l1.txt"
test_tgt="data/test.en-$l1.en.txt"
vocab="data/vocab.$l1-en.bin"

work_dir="work_dir.$l1-en"

echo decoding $dev_src ...
python hypernmt.py \
    decode \
    --beam-size ${beam_size} \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${dev_src} \
    ${work_dir}/decode.dev.beam$beam_size.txt

perl multi-bleu.perl ${dev_tgt} < ${work_dir}/decode.dev.beam$beam_size.txt

echo decoding $test_src ...
python hypernmt.py \
    decode \
    --beam-size ${beam_size} \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.test.beam$beam_size.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.test.beam$beam_size.txt
