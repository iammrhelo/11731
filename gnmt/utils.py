import math
import xml.etree.ElementTree as ET
from typing import List

import numpy as np


def input_transpose(sents, pad_token):
    """
    This function transforms a list of sentences of shape (batch_size, token_num) into
    a list of shape (token_num, batch_size). You may find this function useful if you
    use pytorch
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) >
                        i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        # Sort decreasing by source sentence length
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents, tgt_sents = list(zip(*examples))
        yield src_sents, tgt_sents


def read_iwslt_corpus(file_path, source):
    """ Reads IWSLT Corpus with keywords and language information
    Format example:
    <keywords>kw1, kw2, kw3</keywords>\ten\tHello World .\n
    Returns:
        datas: list of tuples 
        tuple: (keywords, code, sent)
    """
    datas = []
    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            line = line.strip()
            # Split tab
            keyword_raw, code, sentence = line.split('\t')

            # Process keyword
            keyword_node = ET.fromstring(keyword_raw)
            keywords = keyword_node.text.split(', ')
            keywords = list(map(lambda x: '<' + x + '>', keywords))

            # Process code
            assert code in ["en", "de", "nl"]
            code = '<2' + code + '>'

            # Process sentence
            sent = sentence.split()
            if source == "tgt":
                sent = ['<s>'] + sent + ['</s>']
            tup = (keywords, code, sent)
            datas.append(tup)
    return datas
