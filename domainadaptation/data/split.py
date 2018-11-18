"""
Usage:
    python split.py DE_FILE EN_FILE SIZE 

Example:
    python split.py it.de it.en 5000

Outputs:
    it.train.de it.test.de
"""
import os
import random
import sys

# Seed rng
random.seed(11731)

de_file = sys.argv[1]
en_file = sys.argv[2]

size = int(sys.argv[3])


# Read
def read_sents(filepath):
    sents = []
    with open(filepath, 'r') as fin:
        for line in fin.readlines():
            s = line.strip()
            if len(s):
                sents.append(s)
    return sents


de_sents = read_sents(de_file)
en_sents = read_sents(en_file)
assert len(de_sents) == len(en_sents)


# Shuffle
paired_sents = list(zip(de_sents, en_sents))
random.shuffle(paired_sents)
de_sents, en_sents = zip(*paired_sents)

# Split by size
de_train = de_sents[:-size]
de_test = de_sents[-size:]

en_train = en_sents[:-size]
en_test = en_sents[-size:]

# Write to output
filename = os.path.basename(de_file)
domain, lang = filename.split('.')


def write_sents(sents, filepath):
    with open(filepath, 'w') as fout:
        for s in sents:
            fout.write(s + '\n')


write_sents(de_train, domain + ".train.de")
write_sents(de_test, domain + ".test.de")
write_sents(en_train, domain + ".train.en")
write_sents(en_test, domain + ".test.en")
