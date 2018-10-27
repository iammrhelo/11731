
#
#!/bin/bash

lowresource=('az' 'be' 'gl');
highresource=('tr' 'ru' 'pt');

cd ..
for ((idx=0; idx<${#lowresource[@]}; ++idx)); 
do
    l1=${lowresource[idx]}; 
    l2=${highresource[idx]};
    echo Processing $l1+$l2
    echo "python vocab.py --train-src=./data.bpe/train.en-$l1+$l2.$l1+$l2.bpe.txt --train-tgt=./data.bpe/train.en-$l1+$l2.en.bpe.txt ./data.bpe/vocab.$l1+$l2-en.bpe.bin"
    python vocab.py --train-src=./data.bpe/train.en-$l1+$l2.$l1+$l2.bpe.txt --train-tgt=./data.bpe/train.en-$l1+$l2.en.bpe.txt ./data.bpe/vocab.$l1+$l2-en.bpe.bin
done;

