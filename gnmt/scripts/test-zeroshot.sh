#!/bin/sh

name='nl-en-de'
test_name='de-nl'

in_dir='../iwslt2017/normal'
out_dir='../iwslt2017/data'

dev_src=${out_dir}/valid.${name}.src
dev_tgt=${out_dir}/valid.${name}.tgt

work_dir=work_dir-${name}
beam_size=3
echo decoding $dev_src ...
python gnmt_skeleton.py \
    decode \
    --beam-size ${beam_size} \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${dev_src} \
    ${work_dir}/decode.dev.beam$beam_size.txt

perl multi-bleu.perl ${dev_tgt} < ${work_dir}/decode.dev.beam$beam_size.txt


IFS="-" read -ra langs <<< "${test_name}";
echo ${langs[0]} ${langs[1]}

test_src=${in_dir}/test.${test_name}.${langs[0]}
test_tgt=${in_dir}/test.${test_name}.${langs[1]}

echo decoding $test_src ...
python gnmt_skeleton.py \
    decode \
    --beam-size ${beam_size} \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.test.beam$beam_size.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.${name}.test.beam$beam_size.txt
