#!/bin/sh

in_dir='data'
out_dir='data'

beam_size=5

names=('de-nl-en' 'nl-en-de' 'en-de-nl');
test_pairs=('en-de' 'de-nl' 'nl-en');

for ((idx=0; idx<${#names[@]}; ++idx)); 
do
    name=${names[idx]}; 
    test_name=${test_pairs[idx]};

    work_dir=work_dir2-${name}
    IFS="-" read -ra langs <<< "${test_name}";
    echo ${langs[0]} ${langs[1]}

    test_src=${in_dir}/test.${langs[0]}-${langs[1]}.${langs[0]}
    test_tgt=${in_dir}/test.${langs[0]}-${langs[1]}.${langs[1]}

    echo decoding $test_src ...
    python gnmt_skeleton.py \
        --cuda \
        decode \
        --beam-size ${beam_size} \
        --max-decoding-time-step 100 \
        ${work_dir}/model.bin \
        ${test_src} \
        ${test_tgt} \
        ${work_dir}/decode.${langs[0]}-${langs[1]}.test.beam$beam_size.txt

    #perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.${name}.test.beam$beam_size.txt


    test_src=${in_dir}/test.${langs[0]}-${langs[1]}.${langs[1]}
    test_tgt=${in_dir}/test.${langs[0]}-${langs[1]}.${langs[0]}

    echo decoding $test_src ...
    python gnmt_skeleton.py \
        --cuda \
        decode \
        --beam-size ${beam_size} \
        --max-decoding-time-step 100 \
        ${work_dir}/model.bin \
        ${test_src} \
        ${test_tgt} \
        ${work_dir}/decode.${langs[1]}-${langs[0]}.test.beam$beam_size.txt

    #perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.${name}.test.beam$beam_size.txt
done;
