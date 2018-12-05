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

lang_pairs=('nl-en' 'en-de' 'de-nl');

for ((idx=0; idx<${#lang_pairs[@]}; ++idx)); 
do
    pair=${lang_pairs[idx]}; 
    echo processing $pair
    
    IFS="-" read -ra langs <<< "${pair}";
    echo ${langs[0]} ${langs[1]}

    test_src=${in_dir}/test.${pair}.${langs[0]}
    test_tgt=${in_dir}/test.${pair}.${langs[1]}

    echo decoding $test_src ...
    python gnmt_skeleton.py \
        decode \
        --beam-size ${beam_size} \
        --max-decoding-time-step 100 \
        ${work_dir}/model.bin \
        ${test_src} \
        ${test_tgt} \
        ${work_dir}/decode.${pair}.test.beam$beam_size.txt

    #perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.${pair}.test.beam$beam_size.txt

done;

