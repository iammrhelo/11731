#!/bin/sh

vocab="debug/vocab.debug.bin"
# train_src="data/train.de-en.de.wmixerprep"
# train_tgt="data/train.de-en.en.wmixerprep"
dev_src="debug/valid.en-de.de"
dev_tgt="debug/valid.en-de.en"
test_src="debug/test.en-de.de"
test_tgt="debug/test.en-de.en"

work_dir="work_dir.debug"
beam_size=1
echo decoding $dev_src ...
/remote/bones/user/dspokoyn/anaconda3/bin/python3.7 cpg_skeleton.py \
    decode \
    --beam-size ${beam_size} \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${dev_src} \
    ${work_dir}/decode.dev.beam$beam_size.txt\
    --cuda

perl multi-bleu.perl ${dev_tgt} < ${work_dir}/decode.dev.beam$beam_size.txt

# echo decoding $test_src ...
# /remote/bones/user/dspokoyn/anaconda3/bin/python3.7 cpg_skeleton.py \
#     decode \
#     --beam-size ${beam_size} \
#     --max-decoding-time-step 100 \
#     ${work_dir}/model.bin \
#     ${test_src} \
#     ${work_dir}/decode.test.beam$beam_size.txt

# perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.test.beam$beam_size.txt
