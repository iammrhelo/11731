#!/bin/sh

in_dir='../iwslt2017/normal'
out_dir='../iwslt2017/data'

beam_size=5

names=('nl-en-de' 'en-de-nl' 'de-nl-en');

for ((idx=0; idx<${#names[@]}; ++idx)); 
do
    name=${names[idx]}; 
    work_dir=work_dir-${name}

    dev_src=${out_dir}/valid.${name}.src
    dev_tgt=${out_dir}/valid.${name}.tgt

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

done;