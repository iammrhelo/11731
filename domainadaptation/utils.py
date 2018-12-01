import math
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

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

def zeroshot_batch_iter(data, batch_size, shuffle=False):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    print(batch_num)
    fooo
    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]
        src_langs = [e[2] for e in examples]
        tgt_langs = [e[3] for e in examples]

        yield src_sents, tgt_sents, src_langs, tgt_langs

def hyper_read_corpus(file_path, source):
    datas = []
    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            domain_file_path = line.strip()
            domain_data = read_corpus(domain_file_path, source)
            datas.append(domain_data)
    return datas


def hyper_zeroshot_batch_iter(datas, batch_size, shuffle=False):
    """
    Hyper Batch Iterator, implements the following sampling strategies
    Only used in training

    1. Randomly sample language pair and randomly sample a batch from that language pair => no epochs
    2. Randomly sample languages pairs until all train data is visited => epoch

    Args:
        datas: list of domain data
    """
    assert shuffle == True
    lang_pairs = len(datas)

    # Create generators
    lang_pair_iters = [zeroshot_batch_iter(
        data, batch_size, shuffle) for data in datas]

    while True:

        idx = np.random.choice(lang_pairs)

        # Yield next batch
        try:
            src_sents, tgt_sents, src_langs, tgt_langs = next(lang_pair_iters[idx])
            fooo
        except StopIteration:
            # Recreate generator
            lang_pair_iters[idx] = zeroshot_batch_iter(datas[idx], batch_size, shuffle)
            src_sents, tgt_sents, src_langs, tgt_langs = next(lang_pair_iters[idx])
        finally:
            yield idx, (src_sents, tgt_sents, src_langs, tgt_langs)


def hyper_batch_iter(datas, batch_size, shuffle=False):
    """
    Hyper Batch Iterator, implements the following sampling strategies
    Only used in training

    1. Randomly sample language pair and randomly sample a batch from that language pair => no epochs
    2. Randomly sample languages pairs until all train data is visited => epoch

    Args:
        datas: list of domain data
    """
    assert shuffle == True
    domain_size = len(datas)

    # Create generators
    domain_iters = [batch_iter(
        data, batch_size, shuffle) for data in datas]

    while True:

        idx = np.random.choice(domain_size)

        # Yield next batch
        try:
            src_sents, tgt_sents = next(domain_iters[idx])
        except StopIteration:
            # Recreate generator
            domain_iters[idx] = batch_iter(datas[idx], batch_size, shuffle)
            src_sents, tgt_sents = next(domain_iters[idx])
        finally:
            yield idx, (src_sents, tgt_sents)
