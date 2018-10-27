#!/bin/bash
# Apply Bye Pair Encodings to both source and target combined pairs

lowresource=('az' 'be' 'gl');
highresource=('tr' 'ru' 'pt');

num_ops=10000

data_dir=./data
bpe_dir=./data.bpe

for ((idx=0; idx<${#lowresource[@]}; ++idx)); 
do
    l1=${lowresource[idx]};
    l2=${highresource[idx]};
    echo Learning joint BPE with $l1 and $l2...

    train_file=$data_dir/train.en-$l1+$l2.$l1+$l2.txt
    codecs_file=$bpe_dir/codecs.en-$l1+$l2.$l1+$l2.bpe.txt
    out_file=$bpe_dir/train.en-$l1+$l2.$l1+$l2.bpe.txt
    
    subword-nmt learn-bpe -v -s ${num_ops} < $train_file > $codecs_file 

    # Apply BPE
    echo applying to train...
    subword-nmt apply-bpe -c $codecs_file < $train_file > $out_file

    for split in dev test;
    do
        echo applying to $split...
        split_file=$data_dir/$split.en-$l1.$l1.txt
        out_file=$bpe_dir/$split.en-$l1.$l1.bpe.txt
        subword-nmt apply-bpe -c $codecs_file < $split_file > $out_file
    done;

    echo Learning BPE with English counterpart...

    train_file=$data_dir/train.en-$l1+$l2.en.txt
    codecs_file=$bpe_dir/codecs.en-$l1+$l2.en.bpe.txt
    out_file=$bpe_dir/train.en-$l1+$l2.en.bpe.txt
    
    subword-nmt learn-bpe -v -s ${num_ops} < $train_file > $codecs_file 

    # Apply Byte Pair Encoding
    for split in train dev test;
    do
        echo applying to $split...
        split_file=$data_dir/$split.en-$l1.en.txt
        out_file=$bpe_dir/$split.en-$l1.en.bpe.txt
        subword-nmt apply-bpe -c $codecs_file < $split_file > $out_file
    done;

 done;

