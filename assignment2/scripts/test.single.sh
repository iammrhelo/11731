#!/bin/sh

l1=$1
beam_size=$2

data_dir=./data

vocab="data/vocab.$l1-en.bin"
work_dir="work_dir.$l1-en"
src_postfix=".en-$l1.$l1.txt"
tgt_postfix=".en-$l1.en.txt"

for split in dev test;
do 
    echo decoding ${split}_src ...
    src_file=${data_dir}/${split}${src_postfix}
    tgt_file=${data_dir}/${split}${tgt_postfix}

    dec_file=${work_dir}/decode.${split}.beam$beam_size.txt 

    echo decoding $src_file to $dec_file 

    python nmt.py \
        decode \
        --beam-size ${beam_size} \
        --max-decoding-time-step 100 \
        ${work_dir}/model.bin \
        ${src_file} \
        ${dec_file}

    perl multi-bleu.perl ${tgt_file} < ${dec_file}
done;   
