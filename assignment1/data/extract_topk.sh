k=2
for f in *;
do  
    head -$k $f > ../data.debug/$f
    echo $f;
done;