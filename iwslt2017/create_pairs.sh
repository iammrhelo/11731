#!/bin/bash

# A example to produce whole training data 
# cat IN_FILE >>> OUT_FILE
# appends contents of IN_FILE to OUT_FILE

split="train"
# creates "train.src and train.tgt

# Source     Target
# English -> German
# German  -> English
cat $split.en-de.en >> $split.src 
cat $split.en-de.de >> $split.tgt 
cat $split.en-de.en >> $split.tgt
cat $split.en-de.de >> $split.src 

# Source     Target
# German  -> Dutch
# Dutch   -> German
cat $split.de-nl.de >> $split.src 
cat $split.de-nl.nl >> $split.tgt 
cat $split.de-nl.de >> $split.tgt
cat $split.de-nl.nl >> $split.src 

# Source     Target
# Dutch   -> English
# English -> Dutch
cat $split.nl-en.nl >> $split.src 
cat $split.nl-en.en >> $split.tgt 
cat $split.nl-en.nl >> $split.tgt
cat $split.nl-en.en >> $split.src 
