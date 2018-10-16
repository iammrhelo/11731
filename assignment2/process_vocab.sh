
#
#!/bin/bash

lowresource=('az' 'be' 'gl');
highresource=('tr' 'ru' 'pt');

for ((idx=0; idx<${#lowresource[@]}; ++idx)); 
do
    l1=${lowresource[idx]}; 
    l2=${highresource[idx]};
    echo Processing $l1+$l2
    echo "python vocab.py --train-src=./data/train.en-$l1+$l2.$l1+$l2.txt --train-tgt=./data/train.en-$l1+$l2.en.txt ./data/vocab.$l1+$l2-en.bin"
    python vocab.py --train-src=./data/train.en-$l1+$l2.$l1+$l2.txt --train-tgt=./data/train.en-$l1+$l2.en.txt ./data/vocab.$l1+$l2-en.bin
done;

