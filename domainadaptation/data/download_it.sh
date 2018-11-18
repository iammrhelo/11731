

domain=it

gnome_link="http://opus.nlpl.eu/download.php?f=GNOME/de-en.txt.zip"
kde_link="http://opus.nlpl.eu/download.php?f=KDE4/de-en.txt.zip"
php_link="http://opus.nlpl.eu/download.php?f=PHP/de-en.txt.zip"
ubuntu_link="http://opus.nlpl.eu/download.php?f=Ubuntu/de-en_AU.txt.zip"
openoffice_link="http://opus.nlpl.eu/download.php?f=OpenOffice3/de-en_GB.txt.zip"

mkdir -p $domain

links=($gnome_link $kde_link $php_link $ubuntu_link $openoffice_link)
names=(gnome kde php ubuntu openoffice)

echo ${links[0]}
echo ${names[0]}

tLen=${#names[@]}

for (( i=0; i<${tLen}; i++ )); do
    name=${names[i]}
    link=${links[i]}
    echo $name $link
    # Download
    if [ ! -f $domain/$name.zip ]; then
        echo Start download.
        wget -O $domain/$name.zip $link
    else
        echo $domain/$name.zip already exists. Download skipped.
    fi
    # Extract only new files
    echo Extracing $domain/$name.zip 
    unzip -n $domain/$name.zip -d $domain
done

# Concatenate all files into one big file
de_txt=$domain/all.raw.de
en_txt=$domain/all.raw.en

cat $(ls $domain/*.de) > $de_txt
cat $(ls $domain/*.en) > $en_txt

echo Tokenizing German file $de_txt to $domain.de
perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < $de_txt > $domain.de 
echo Tokenizing English file $en_txt to $domain.en
perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < $en_txt > $domain.en