#
#!/bin/bash

names=('nl-en-de');

in_dir='../iwslt2017/zeroshot'
out_dir='../iwslt2017/data'

for ((idx=0; idx<${#names[@]}; ++idx)); 
do
    name=${names[idx]}; 
    echo $name
    
    IFS="-" read -ra langs <<< "${name}";
    echo ${langs[0]} ${langs[1]} ${langs[2]}

    rm -f $out_dir/train2.$name.src
    rm -f $out_dir/train2.$name.tgt

    rm -f $out_dir/valid2.$name.src
    rm -f $out_dir/valid2.$name.tgt

    cat $in_dir/train1.en-de-nl.${langs[0]} $in_dir/train1.en-de-nl.${langs[1]} >> $out_dir/train2.$name.src
    cat $in_dir/train1.en-de-nl.${langs[1]} $in_dir/train1.en-de-nl.${langs[0]} >> $out_dir/train2.$name.tgt

    cat $in_dir/train2.en-de-nl.${langs[1]} $in_dir/train2.en-de-nl.${langs[2]} >> $out_dir/train2.$name.src
    cat $in_dir/train2.en-de-nl.${langs[2]} $in_dir/train2.en-de-nl.${langs[1]} >> $out_dir/train2.$name.tgt

    cat $in_dir/valid.en-de-nl.${langs[0]} $in_dir/valid.en-de-nl.${langs[1]} >> $out_dir/valid2.$name.src
    cat $in_dir/valid.en-de-nl.${langs[1]} $in_dir/valid.en-de-nl.${langs[0]} >> $out_dir/valid2.$name.tgt

    cat $in_dir/valid.en-de-nl.${langs[1]} $in_dir/valid.en-de-nl.${langs[2]} >> $out_dir/valid2.$name.src
    cat $in_dir/valid.en-de-nl.${langs[2]} $in_dir/valid.en-de-nl.${langs[1]} >> $out_dir/valid2.$name.tgt

done;

