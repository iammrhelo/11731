#
#!/bin/bash

train_pairs=('nl-en' 'en-de');
name='nl-en-de'

in_dir='../iwslt2017/normal'
out_dir='../iwslt2017/data'

for split in train valid;
do
    echo $split
    
    rm $out_dir/$split.$name.src
    rm $out_dir/$split.$name.tgt

    for ((idx=0; idx<${#train_pairs[@]}; ++idx)); 
    do
        pair=${train_pairs[idx]}; 
        echo Processing $pair
        
        IFS="-" read -ra langs <<< "${pair}";
        echo ${langs[0]} ${langs[1]}

        cat $in_dir/$split.$pair.${langs[0]} $in_dir/$split.$pair.${langs[1]} >> $out_dir/$split.$name.src
        cat $in_dir/$split.$pair.${langs[1]} $in_dir/$split.$pair.${langs[0]} >> $out_dir/$split.$name.tgt
    done;
done;

