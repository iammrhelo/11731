
#
#!/bin/bash

lowresource=('az' 'be' 'gl');
highresource=('tr' 'ru' 'pt');

num_ops=10000

for ((idx=0; idx<${#lowresource[@]}; ++idx)); 
do
    l1=${lowresource[idx]};
    l2=${highresource[idx]};
    echo Learning joint BPE with $l1 and $l2...

    train_file=train.en-$l1+$l2.$l1+$l2.txt
    codecs_file=codecs.en-$l1+$l2.bpe.txt
    vocab_file=vocab.$l1+$l2-en.bpe.bin

    out_file=train.en-$l1+$l2.$l1+$l2.bpe.txt
    
    subword-nmt learn-bpe -v -s ${num_ops} < $train_file > $codecs_file 

    # Apply BPE
    echo train
    subword-nmt apply-bpe -c $codecs_file < $train_file > $out_file

    for split in dev test;
    do
        echo $split
        split_file=$split.en-$l1.$l1.txt
        out_file=$split.en-$l1.$l1.bpe.txt
        subword-nmt apply-bpe -c $codecs_file < $split_file > $out_file
    done;

 done;

