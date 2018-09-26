for f in *; do
    echo $f
    head -10 $f > $f.debug
done
