
#
#!/bin/bash

lang_pairs=('nl-en' 'en-de' 'de-nl');
in_dir='../iwslt2017/normal'
out_dir='../iwslt2017/data'

for split in train valid test;
do
    echo $split

    for ((idx=0; idx<${#lang_pairs[@]}; ++idx)); 
    do
        pair=${lang_pairs[idx]}; 
        echo Processing $pair
        
        IFS="-" read -ra langs <<< "${pair}";
        echo ${langs[0]} ${langs[1]}

        cat $in_dir/$split.$pair.${langs[0]} $in_dir/$split.$pair.${langs[1]} >> $out_dir/$split.all.src
        cat $in_dir/$split.$pair.${langs[1]} $in_dir/$split.$pair.${langs[0]} >> $out_dir/$split.all.tgt
    done;
done;

