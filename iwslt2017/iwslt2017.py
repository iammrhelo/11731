"""Process IWSLT2017 dataset
Usage:
    iwslt2017.py split zeroshot
    iwslt2017.py zeroshot
    iwslt2017.py train
    iwslt2017.py valid
    iwslt2017.py test

Options:
    -h --help   Show this message.
"""
import os
import sys
import xml.etree.ElementTree as ET
import random
from glob import glob

from docopt import docopt
from sacremoses import MosesTokenizer
from tqdm import tqdm

random.seed(11731)

en_moses = MosesTokenizer("en")
de_moses = MosesTokenizer("de")
nl_moses = MosesTokenizer("nl")

EN = "en"
DE = "de"
NL = "nl"

DATA_DIR = "DeEnItNlRo-DeEnItNlRo"

ZEROSHOT_DIR = "./zeroshot"
NORMAL_DIR = "./normal"


def read_corpus(file_path):
    sents = []
    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            s = line.strip()
            sents.append(s)
    return sents


def read_train_corpus(file_path, code):
    """Returns keyword tag and sentence pairs
    ("<keywords> ... </keywords>", "..." )
    """
    sents = []
    keywords = None
    print("Reading {}".format(file_path))
    with open(file_path, 'r') as fin:
        for line in tqdm(fin.readlines()):
            line = line.strip()
            if line.startswith("<"):
                if line.startswith("<keywords>"):
                    keywords = line
                continue

            assert keywords is not None

            if code == "en":
                tokens = en_moses.tokenize(line)
            elif code == "de":
                tokens = de_moses.tokenize(line)
            elif code == "nl":
                tokens = nl_moses.tokenize(line)
            else:
                raise Exception

            line = " ".join(tokens)

            s = keywords + "\t" + code + "\t" + line  # Add tab to separate the two
            sents.append(s)
    print("Number of pairs: {}".format(len(sents)))
    return sents


def read_xml_corpus(file_path, code):
    sents = []

    print("Reading {}".format(file_path))
    tree = ET.parse(file_path)
    root = tree.getroot()
    srcset = root[0]

    for doc in srcset:
        assert doc.tag == "doc"
        keywords = None
        for ele in doc:
            if ele.tag == "keywords":
                keywords = "<keywords>" + ele.text + "</keywords>"
                continue
            elif ele.tag == "seg":
                line = ele.text
                if code == "en":
                    tokens = en_moses.tokenize(line)
                elif code == "de":
                    tokens = de_moses.tokenize(line)
                elif code == "nl":
                    tokens = nl_moses.tokenize(line)
                else:
                    raise Exception

                line = " ".join(tokens)
                s = keywords + "\t" + code + "\t" + line
                sents.append(s)
    print("Number of pairs: {}".format(len(sents)))
    return sents


def process_zeroshot():
    # Read all English pairs
    template = os.path.join(DATA_DIR, "train.tags.{}-{}.{}")

    # Read English-German
    en_de_en_file = template.format(EN, DE, EN)
    en_de_de_file = template.format(EN, DE, DE)

    en_de_en_sents = read_train_corpus(en_de_en_file, EN)
    en_de_de_sents = read_train_corpus(en_de_de_file, DE)

    de_en_sents = {}
    filtered_en_de_de_sents = []
    for en, de in zip(en_de_en_sents, en_de_de_sents):
        en_tokens = en.split('\t')[-1].split()
        de_tokens = de.split('\t')[-1].split()
        if len(en_tokens) <= 50 and len(de_tokens) <= 50:
            de_en_sents[de] = en
            filtered_en_de_de_sents.append(de)

    print("EN-DN pairs: ", len(de_en_sents))

    # Read German-Dutch
    de_nl_de_file = template.format(DE, NL, DE)
    de_nl_nl_file = template.format(DE, NL, NL)

    de_nl_de_sents = read_train_corpus(de_nl_de_file, DE)
    de_nl_nl_sents = read_train_corpus(de_nl_nl_file, NL)

    de_nl_sents = {}
    filtered_de_nl_de_sents = []
    for de, nl in zip(de_nl_de_sents, de_nl_nl_sents):
        de_tokens = de.split('\t')[-1].split()
        nl_tokens = nl.split('\t')[-1].split()
        if len(de_tokens) <= 50 and len(nl_tokens) <= 50:
            de_nl_sents[de] = nl
            filtered_de_nl_de_sents.append(de)

    print("De-NL pairs: ", len(de_nl_sents))

    # German sentence overlap
    en_de_de_nl_overlap_de = set(
        filtered_en_de_de_sents) & set(filtered_de_nl_de_sents)
    print("Overlap De from EN-DE & DE-NL:", len(en_de_de_nl_overlap_de))

    # Triples from En-DE & DE-NL
    en_de_nl_triplets = []
    for de in en_de_de_nl_overlap_de:
        en = de_en_sents[de]
        nl = de_nl_sents[de]
        en_de_nl_triplets.append((en, de, nl))

    print("Triplets from en-de and de-nl", len(en_de_nl_triplets))

    # Read NL-EN
    nl_en_nl_file = template.format(NL, EN, NL)
    nl_en_en_file = template.format(NL, EN, EN)

    nl_en_nl_sents = read_train_corpus(nl_en_nl_file, NL)
    nl_en_en_sents = read_train_corpus(nl_en_en_file, EN)
    nl_en_sents = {nl: en for nl, en in zip(nl_en_nl_sents, nl_en_en_sents)}
    print("NL-EN pairs: ", len(nl_en_sents))

    final_triplets = []

    for en, de, nl in en_de_nl_triplets:
        if de not in de_nl_sents:
            continue
        final_triplets.append((en, de, nl))

    print("Final triples", len(final_triplets))
    # Save to output file
    if not os.path.exists(ZEROSHOT_DIR):
        os.makedirs(ZEROSHOT_DIR)

    en_sents, de_sents, nl_sents = list(zip(*final_triplets))

    # Tokenize with moses
    en_path = os.path.join(ZEROSHOT_DIR, "zeroshot.en-de-nl.en")
    de_path = os.path.join(ZEROSHOT_DIR, "zeroshot.en-de-nl.de")
    nl_path = os.path.join(ZEROSHOT_DIR, "zeroshot.en-de-nl.nl")

    write_to_file(en_sents, en_path)
    write_to_file(de_sents, de_path)
    write_to_file(nl_sents, nl_path)


def process_normal(opt):
    if opt["train"]:
        template = os.path.join(DATA_DIR, "train.tags.{}-{}.{}")
        split = "train"
        read_fnt = read_train_corpus
    elif opt["valid"]:
        template = os.path.join(DATA_DIR, "IWSLT17.TED.dev2010.{}-{}.{}.xml")
        split = "valid"
        read_fnt = read_xml_corpus
    elif opt["test"]:
        template = os.path.join(DATA_DIR, "IWSLT17.TED.tst2010.{}-{}.{}.xml")
        split = "test"
        read_fnt = read_xml_corpus

    # Read and process 3 language pairs
    # Read English-German
    code_pairs = [(EN, DE), (DE, NL), (NL, EN)]

    for c1, c2 in code_pairs:

        c1_c2_c1_file = template.format(c1, c2, c1)
        c1_c2_c2_file = template.format(c1, c2, c2)

        c1_c2_c1_sents = read_fnt(c1_c2_c1_file, c1)
        c1_c2_c2_sents = read_fnt(c1_c2_c2_file, c2)

        tmp1 = []
        tmp2 = []
        for s1, s2 in zip(c1_c2_c1_sents, c1_c2_c2_sents):
            token1 = s1.split('\t')[-1].split()
            token2 = s1.split('\t')[-1].split()
            if split == "train" and len(token1) <= 50 and len(token2) <= 50:
                tmp1.append(s1)
                tmp2.append(s2)

        c1_c2_c1_sents = tmp1
        c1_c2_c2_sents = tmp2

        assert len(c1_c2_c1_sents) == len(c1_c2_c2_sents)
        print("{}-{} pairs: {}".format(c1, c2, len(c1_c2_c1_sents)))

        c1_c2_c1_out = os.path.join(
            NORMAL_DIR, "{}.{}-{}.{}".format(split, c1, c2, c1))
        c1_c2_c2_out = os.path.join(
            NORMAL_DIR, "{}.{}-{}.{}".format(split, c1, c2, c2))

        write_to_file(c1_c2_c1_sents, c1_c2_c1_out)
        write_to_file(c1_c2_c2_sents, c1_c2_c2_out)


def write_to_file(sents, file_path):
    with open(file_path, 'w') as fout:
        for s in sents:
            assert isinstance(s, str)
            s = s.strip()
            fout.write(s + "\n")


def split_zeroshot(opt):
    # Split train

    en_sents = read_corpus(os.path.join(
        ZEROSHOT_DIR, "zeroshot.{}-{}-{}.{}".format(EN, DE, NL, EN)))
    de_sents = read_corpus(os.path.join(
        ZEROSHOT_DIR, "zeroshot.{}-{}-{}.{}".format(EN, DE, NL, DE)))
    nl_sents = read_corpus(os.path.join(
        ZEROSHOT_DIR, "zeroshot.{}-{}-{}.{}".format(EN, DE, NL, NL)))

    assert len(en_sents) == len(de_sents) and len(de_sents) == len(nl_sents)

    all_sents = list(zip(en_sents, de_sents, nl_sents))

    random.shuffle(all_sents)

    train_sents = all_sents[:-2000]
    valid_sents = all_sents[-2000:-1000]
    test_sents = all_sents[-1000:]

    train_split = int(len(train_sents)/2)

    train_1_sents = train_sents[:train_split]
    train_2_sents = train_sents[-train_split:]

    def split_and_write(sents, split):

        en_sents, de_sents, nl_sents = list(zip(*sents))

        en_path = os.path.join(
            ZEROSHOT_DIR, "{}.en-de-nl.{}".format(split, EN))
        de_path = os.path.join(
            ZEROSHOT_DIR, "{}.en-de-nl.{}".format(split, DE))
        nl_path = os.path.join(
            ZEROSHOT_DIR, "{}.en-de-nl.{}".format(split, NL))

        write_to_file(en_sents, en_path)
        write_to_file(de_sents, de_path)
        write_to_file(nl_sents, nl_path)

    split_and_write(train_1_sents, "train1")
    split_and_write(train_2_sents, "train2")
    split_and_write(valid_sents, "valid")
    split_and_write(test_sents, "test")


if __name__ == "__main__":
    opt = docopt(__doc__)

    if opt["split"] and opt["zeroshot"]:
        split_zeroshot(opt)
    elif opt["zeroshot"]:
        process_zeroshot()
    elif opt["train"] or opt["valid"] or opt["test"]:
        process_normal(opt)
