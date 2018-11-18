#!/bin/bash
# Download data from all domains
# 1. Law(Acquis) 2. Medical(EMEA) 3. IT(GNOME, KDE, PHP, Ubuntu, OpenOffice) 4. Koran(Tanzil) 5. Subtitles
#  Domain Options
# law | medical | it | koran | subtitles
domain=$1

if [ ! -d "mosesdecoder" ]; then
    # Control will enter here if $DIRECTORY doesn't exist.
    git clone https://github.com/moses-smt/mosesdecoder
fi

# Set links to download
echo Input Domain: $domain
if [ "$domain" == "law" ]; then
    link="http://opus.nlpl.eu/download.php?f=JRC-Acquis/de-en.txt.zip"
elif [ "$domain" == "medical" ]; then
    link="http://opus.nlpl.eu/download.php?f=EMEA/de-en.txt.zip"
elif [ "$domain" == "it" ]; then
    echo Hello World
elif [ "$domain" == "koran" ]; then
    link="http://opus.nlpl.eu/download.php?f=Tanzil/de-en.txt.zip"
elif [ "$domain" == "subtitles" ]; then
    link="http://opus.nlpl.eu/download.php?f=OpenSubtitles2018/de-en.txt.zip"
else
    echo Unknown domain: $domain
    exit 1 
fi 

# Make directory
mkdir -p $domain

# Download 
if [ ! -f $domain/download.zip ]; then
    echo Start download.
    wget -O $domain/download.zip $link
else
    echo $domain/download.zip already exists. Download skipped.
fi

# Extract only new files
unzip -f $domain/download.zip -d $domain

de_txt=$(ls $domain/*.de)
en_txt=$(ls $domain/*.en)

echo English file: $en_txt

# Tokenize with moses 
echo Tokenizing German file $de_txt to $domain.de
perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < $de_txt > $domain.de 
echo Tokenizing English file $en_txt to $domain.en
perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < $en_txt > $domain.en