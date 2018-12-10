#!/bin/sh

in_dir='../iwslt2017/normal'
out_dir='../iwslt2017/data'

dev_src=${out_dir}/valid.all.src
dev_tgt=${out_dir}/valid.all.tgt

work_dir="work_dir-all"
beam_size=5

echo decoding $dev_src ...
python gnmt_skeleton.py \
    decode \
    --beam-size ${beam_size} \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${dev_src} \
    ${dev_tgt} \
    ${work_dir}/decode.dev.beam$beam_size.txt

#perl multi-bleu.perl ${dev_tgt} < ${work_dir}/decode.dev.beam$beam_size.txt