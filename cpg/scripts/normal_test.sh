#!/bin/sh

vocab="normal/vocab.normal.bin"
dev_src_list="normal/normal_valid_src_list.txt"
dev_tgt_list="normal/normal_valid_tgt_list.txt"
test_src_list="normal/normal_test_src_list.txt"
test_tgt_list="normal/normal_test_tgt_list.txt"
work_dir="work_dir.normal"

beam_size=5
# echo decoding $dev_src ...
# /remote/bones/user/dspokoyn/anaconda3/bin/python3.7 cpg_skeleton.py \
#     decode \
#     --beam-size ${beam_size} \
#     --max-decoding-time-step 100 \
#     ${work_dir}/model.bin \
#     ${dev_src_list} \
#     ${dev_tgt_list} \
#     ${work_dir}/decode.dev.beam$beam_size.txt\
#     --cuda

# perl multi-bleu.perl ${dev_tgt} < ${work_dir}/decode.dev.beam$beam_size.txt

echo decoding $test_src ...
/remote/bones/user/dspokoyn/anaconda3/bin/python3.7 cpg_skeleton.py \
    decode \
    --beam-size ${beam_size} \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${test_src_list} \
    ${test_tgt_list} \
    ${work_dir}/decode.test.beam$beam_size.txt\
    --cuda

# perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.test.beam$beam_size.txt
