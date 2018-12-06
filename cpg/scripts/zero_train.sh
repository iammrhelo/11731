#!/bin/sh

exp_num=1
vocab="normal/vocab.zero_${exp_num}.bin"
train_src_list="normal/zero_train_${exp_num}_src_list.txt"
train_tgt_list="normal/zero_train_${exp_num}_tgt_list.txt"
dev_src_list="normal/zero_valid_${exp_num}_src_list.txt"
dev_tgt_list="normal/zero_valid_${exp_num}_tgt_list.txt"
work_dir="work_dir.zero${exp_num}"

mkdir -p ${work_dir}
echo save results to ${work_dir}

batch_size=64
b=$batch_size
# valid_niter=20

echo batch_size ${batch_size}
/remote/bones/user/dspokoyn/anaconda3/bin/python3.7 cpg_skeleton.py \
    train \
    --cuda \
    --vocab ${vocab} \
    --train-src ${train_src_list} \
    --train-tgt ${train_tgt_list} \
    --dev-src ${dev_src_list} \
    --dev-tgt ${dev_tgt_list} \
    --save-to ${work_dir} \
    --num-layers 1 \
    --max-epoch 10 \
    --valid-niter ${valid_niter} \
    --batch-size ${batch_size} \
    --hidden-size 50 \
    --embed-size 50 \
    --lang-embed-size 7 \
    --uniform-init 0.1 \
    --dropout 0.0 \
    --clip-grad 5.0 \
    --lr-decay 0.5 \
    --patience 10 \
    --bidirectional  \


#>${work_dir}/err.log
