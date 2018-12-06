#!/bin/sh

exp_num=1
vocab="zeroshot/vocab.hardzero_${exp_num}.bin"
train_src_list="zeroshot/hardzero_train_${exp_num}_src_list.txt"
train_tgt_list="zeroshot/hardzero_train_${exp_num}_tgt_list.txt"
dev_src_list="zeroshot/zero_valid_${exp_num}_src_list.txt"
dev_tgt_list="zeroshot/zero_valid_${exp_num}_tgt_list.txt"
work_dir="work_dir.hardzero${exp_num}"

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
    --max-epoch 30 \
    --batch-size ${batch_size} \
    --hidden-size 256 \
    --embed-size 256 \
    --lang-embed-size 7 \
    --uniform-init 0.1 \
    --dropout 0.0 \
    --clip-grad 5.0 \
    --lr-decay 0.5 \
    --patience 10 \
    --bidirectional  \


#>${work_dir}/err.log
