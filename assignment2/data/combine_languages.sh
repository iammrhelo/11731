
#
#!/bin/bash

lowresource=('az' 'be' 'gl');
highresource=('tr' 'ru' 'pt');


for ((idx=0; idx<${#lowresource[@]}; ++idx)); 
do
    l1=${lowresource[idx]}; 
    l2=${highresource[idx]};
    echo Processing $l1 $l2
    for split in train dev test;
    do
        echo $split
        cat $split.en-$l1.$l1.txt $split.en-$l2.$l2.txt > $split.en-$l1+$l2.$l1+$l2.txt
        cat $split.en-$l1.en.txt $split.en-$l2.en.txt > $split.en-$l1+$l2.en.txt
    done;
done;

