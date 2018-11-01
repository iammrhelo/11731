#!/bin/sh

l1=$1
type=$2

giza_dir=./data.giza/$l1/

name=$type.en-$l1
src_name=$name.$1
tgt_name=$name.en

src_file=$giza_dir$src_name
tgt_file=$giza_dir$tgt_name

echo "create giza++ format files - vcb and snt"
plain2snt.out ${src_file}.txt ${tgt_file}.txt

echo "mkcls"
mkcls -m2 -p ${src_file}.txt -c50 -V ${src_file}.vcb.classes opt > ${src_file}.class.log &
mkcls -m2 -p ${tgt_file}.txt -c50 -V ${tgt_file}.vcb.classes opt > ${tgt_file}.class.log &

echo "run GIZA++"
GIZA++ -S ${src_file}.vcb -T ${tgt_file}.vcb -C ${giza_dir}${src_name}_${tgt_name}.snt -p0 0.98 -o ${giza_dir}${name}.dict > ${giza_dir}${name}.dict.log &





