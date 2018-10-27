
#
#!/bin/bash

lowresource=('az' 'be' 'gl');

num_ops=5000

for ((idx=0; idx<${#lowresource[@]}; ++idx)); 
do
    # Source language
    l1=${lowresource[idx]};
    echo Applying BPE to $l1 ...
    src_file=train.en-$l1.$l1.txt
    src_codecs=train.en-$l1.$l1.codecs
    subword-nmt learn-bpe -s ${num_ops} < $src_file > $src_codecs
    for split in train dev test;
    do
        echo $split
        split_file=$split.en-$l1.$l1.txt
        split_out=$split.en-$l1.$l1.bpe.txt
        subword-nmt apply-bpe -c $src_codecs < $split_file > $split_out
    done;

    echo and English...
    # English
    tgt_file=train.en-$l1.en.txt
    tgt_codecs=train.en-$l1.en.codecs
    subword-nmt learn-bpe -s ${num_ops} < $tgt_file > $tgt_codecs
    for split in train dev test;
    do
        echo $split
        split_file=$split.en-$l1.en.txt
        split_out=$split.en-$l1.en.bpe.txt
        subword-nmt apply-bpe -c $tgt_codecs < $split_file > $split_out
    done;

    # Learn Joint
    #echo and Joint BPE...
    #joint_codecs=$train.en-$l1.joint.codecs 
    #src_vocab=vocab.$l1-en.$l1.joint.bpe.bin
    #tgt_vocab=vocab.$l1-en.en.joint.bpe.bin
    #subword-nmt learn-joint-bpe-and-vocab --input ${src_file} ${tgt_file} -s $num_ops -o $joint_codecs --write-vocabulary $src_vocab $tgt_vocab
    #for split in train dev test;
    #do
    #    split_src_file=$split.en-$l1.$l1.txt
    #    split_src_out=$split.en-$l1.$l1.joint.bpe.txt
    #    subword-nmt apply-bpe -c $joint_codecs --vocabulary $src_vocab < $split_src_file > $split_src_out
    #    split_tgt_file=$split.en-$l1.en.txt
    #    split_tgt_out=$split.en-$l1.en.joint.bpe.txt
    #    subword-nmt apply-bpe -c $joint_codecs --vocabulary $tgt_vocab < $split_tgt_file > $split_tgt_out
    #done;
    


 done;

