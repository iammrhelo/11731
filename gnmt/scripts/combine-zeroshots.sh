#
#!/bin/bash

names=('nl-en-de' 'en-de-nl' 'de-nl-en')

in_dir='data'
out_dir='data'

for ((idx=0; idx<${#names[@]}; ++idx)); 
do
    name=${names[idx]}; 
    echo $name
    
    IFS="-" read -ra langs <<< "${name}";
    echo ${langs[0]} ${langs[1]} ${langs[2]}

    rm $out_dir/train1.$name.src
    rm $out_dir/train1.$name.tgt

    pair=${langs[0]}-${langs[1]};
    echo Processing $pair

    cat $in_dir/train.$pair.${langs[0]} $in_dir/train.$pair.${langs[1]} >> $out_dir/train1.$name.src
    cat $in_dir/train.$pair.${langs[1]} $in_dir/train.$pair.${langs[0]} >> $out_dir/train1.$name.tgt

    pair=${langs[1]}-${langs[2]};
    echo Processing $pair

    cat $in_dir/train.$pair.${langs[1]} $in_dir/train.$pair.${langs[2]} >> $out_dir/train1.$name.src
    cat $in_dir/train.$pair.${langs[2]} $in_dir/train.$pair.${langs[1]} >> $out_dir/train1.$name.tgt


    rm -f $out_dir/train2.$name.src
    rm -f $out_dir/train2.$name.tgt

    cat $in_dir/train1.en-de-nl.${langs[0]} $in_dir/train1.en-de-nl.${langs[1]} >> $out_dir/train2.$name.src
    cat $in_dir/train1.en-de-nl.${langs[1]} $in_dir/train1.en-de-nl.${langs[0]} >> $out_dir/train2.$name.tgt

    cat $in_dir/train2.en-de-nl.${langs[1]} $in_dir/train2.en-de-nl.${langs[2]} >> $out_dir/train2.$name.src
    cat $in_dir/train2.en-de-nl.${langs[2]} $in_dir/train2.en-de-nl.${langs[1]} >> $out_dir/train2.$name.tgt


    rm -f $out_dir/valid.$name.src
    rm -f $out_dir/valid.$name.tgt

    pair=${langs[2]}-${langs[0]};
    echo Processing $pair

    cat $in_dir/valid.$pair.${langs[2]} $in_dir/valid.$pair.${langs[0]} >> $out_dir/valid.$name.src
    cat $in_dir/valid.$pair.${langs[0]} $in_dir/valid.$pair.${langs[2]} >> $out_dir/valid.$name.tgt
done;
