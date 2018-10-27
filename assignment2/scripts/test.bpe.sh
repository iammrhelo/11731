#!/bin/sh

l1=$1
l2=$2
beam_size=$3

data_dir=./data
bpe_dir=./data.bpe

vocab="${bpe_dir}/vocab.$l1+$l2-en.bpe.bin"
work_dir="work_dir.$l1+$l2-en.bpe"
src_postfix=".en-$l1.$l1.bpe.txt"
tgt_postfix=".en-$l1.en.txt"

for split in dev test;
do 
    echo decoding ${split}_src ...
    src_bpe_file=${bpe_dir}/${split}${src_postfix}
    tgt_file=${data_dir}/${split}${tgt_postfix}

    dec_bpe_file=${work_dir}/decode.${split}.beam$beam_size.bpe.txt 
    dec_file=${work_dir}/decode.${split}.beam$beam_size.txt 

    echo decoding $src_bpe_file to $dec_bpe_file 

    python nmt.py \
        decode \
        --beam-size ${beam_size} \
        --max-decoding-time-step 100 \
        ${work_dir}/model.bin \
        ${src_bpe_file} \
        ${dec_bpe_file}

    # sed first...
    # Convert bpe to word
    echo converting $dec_bpe_file to $dec_file ...
    sed -r 's/(@@ )|(@@ ?$)//g' ${dec_bpe_file} > ${dec_file}

    perl multi-bleu.perl ${tgt_file} < ${dec_file}
done;   
