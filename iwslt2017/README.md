# IWSLT 2017 Dataset for NMT Final Project

We selected 3 languages: English(en), German(de), and Dutch(nl).

There are two different types of data
1. normal - normal processing scheme
2. zeroshot - true zero shot language pairs data

To train in a multilingual scenario, you will probably need to concatenate your own data.  My own scenario is to train on en-de, de-nl and test on nl-en, both ways. A script is provided in the final section of this README.

### File Format

In every directory, all file names come as ```SPLIT.L1-L2.L1``` form (e.g., ```train.en-de.en```). In zeroshot directory, this comes as ```SPLIT.L1-L2-L3.L1``` (e.g, train1.en-de-nl.en), since we align 3 languages.

All the files has the following line format, and has 3 types of data.
- Keywords: keywords are separated by comma and has a "keyword" clojure.
- Language code: denotes the language this sentence is in (e.g., en, de, nl)
- Tokenized sentence: space separated sentence

Below is an example
```
<keywords>KEYWORD1 </keywords>\tCODE\tSENTENCE
<keywords>talks, TEDx, communication, technology</keywords>	en	My outbox for email is massive , and I never look at it . I write all the time , but I never look at my record , at my trace .
```

And a python snippet to parse the information
```python
# snippet.py
import xml.etree.ElementTree as ET

with open("./normal/train.en-de.en", "r") as fin:
    for line in fin.readlines():
        # Strip endline
        line = line.strip()
        # Split tab
        keyword_raw, code, sentence = line.split('\t')
        # Get keywords
        keyword_node = ET.fromstring(keyword_raw)
        keywords = keyword_node.text.split(', ')
        # Get code
        assert code in ["en", "de", "nl"]
        # Get individual words
        words = sentence.split()
```

Compared to original format in previous assignments, you will not only the sentence, get the keyword information and the language information for your usage.


### Normal
```./normal``` stores the normal preprocessed version as in CPG.  I used a python port of moses for tokenization, and filtered training data for sentence length less or equal than 50.

We have 3 different language pairs ```en-de```, ```de-nl``` and ```nl-de```.  The splits are ```train```, ```valid```, ```test```.

### Zeroshot

```./zeroshot``` stores using the same processing methods.  However, the splits comes from only the training data in IWSLT2017, though that is still quite a lot. ```zeroshot``` has all the language pairs aligned, no matter in splits ```train1```, ```train2```, ```valid``` or ```test```.

```train1``` and ```train2``` are different splits of the language pairs, each language having around 78k sentences.  ```valid``` and ```test``` each has 1k sentences.


### Multilingual Training File Creation

For multilingual training, you would probably have to concatenate files, depending on whether you want zero-shot (neglecting one pair) or full parallel data(all).  I have a script here ```./create_pairs.sh```, which shows an example.
