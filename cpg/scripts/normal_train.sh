#!/bin/sh

vocab="normal/vocab.normal.bin"
train_src_list="normal/normal_train_src_list.txt"
train_tgt_list="normal/normal_train_tgt_list.txt"
dev_src_list="normal/normal_valid_src_list.txt"
dev_tgt_list="normal/normal_valid_tgt_list.txt"
work_dir="work_dir.normal"

mkdir -p ${work_dir}
echo save results to ${work_dir}

batch_size=64
a=$(wc -l < "${train_src}")
b=$batch_size
valid_niter=$((a%b?a/b+1:a/b))
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
    --max-epoch 1 \
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
