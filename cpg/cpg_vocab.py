#!/usr/bin/env python
"""
Modified vocab.py for IWSLT corpus
Generate the vocabulary file for neural network training
A vocabulary file is a mapping of tokens to their indices

Usage:
    vocab.py --train-src-list=<file> --train-tgt-list=<file> [options] VOCAB_FILE

Options:
    -h --help                       Show this message.
    --train-src-list=<file>         A file containing paths to different files split by language codes
    --train-tgt-list=<file>         A file containings path to different files split by language codes
    --size=<int>                    vocab size [default: 20000]
    --freq-cutoff=<int>             frequency cutoff [default: 5]
"""

from typing import List
from collections import Counter
from itertools import chain
from docopt import docopt
import pickle

from utils import read_iwslt_corpus


class VocabEntry(object):
    def __init__(self, is_sentence=True):
        self.word2id = dict()
        self.unk_id = 0
        self.word2id['<unk>'] = self.unk_id

        if is_sentence:
            self.pad_id = 1
            self.word2id['<pad>'] = self.pad_id
            self.word2id['<s>'] = 2
            self.word2id['</s>'] = 3

        self._id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self._id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self._id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, indices):
        if len(indices) == 0:
            return []
        elif type(indices[0]) == list:
            return [self.indices2words(i) for i in indices]
        else:
            return [self.id2word(i) for i in indices]

    @classmethod
    def from_corpus(cls, corpus, size=None, freq_cutoff=2, is_sentence=True):

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(
            f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}'
        )
        if size:
            top_k_words = sorted(
                valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        else:
            top_k_words = valid_words

        # Create vocab class here
        vocab_entry = cls(is_sentence)
        for word in top_k_words:
            vocab_entry.add(word)

        return vocab_entry


class Vocab(object):
    def __init__(self, src_list, tgt_list, vocab_size, freq_cutoff):
        assert len(src_list) == len(tgt_list)

        # Individual embeddings
        all_codes = set()
        all_keywords = set()

        # Read source files
        print('initialize source vocabulary ..')
        self.src = dict()
        for src_corpus_path in src_list:
            # Read IWSLT corpus
            print('read in source file %s' % src_corpus_path)
            src_data = read_iwslt_corpus(src_corpus_path, source="src")
            src_keywords, src_codes, src_sents = list(zip(*src_data))

            # Process keywords
            for kw in chain(*src_keywords):
                all_keywords.add(kw)

            # Should share the same code
            assert (s == src_codes[0] for s in src_codes)
            code = src_codes[0]
            all_codes.add(code)

            self.src[code] = VocabEntry.from_corpus(
                src_sents, vocab_size, freq_cutoff, is_sentence=True)

        print('initialize target vocabulary ..')
        self.tgt = dict()
        for tgt_corpus_path in tgt_list:
            # Read IWSLT corpus
            print('read in target file %s' % tgt_corpus_path)
            tgt_data = read_iwslt_corpus(tgt_corpus_path, source="tgt")
            tgt_keywords, tgt_codes, tgt_sents = list(zip(*tgt_data))

            # Process keywords
            for kw in chain(*tgt_keywords):
                all_keywords.add(kw)

            # Should share the same code
            assert (t == tgt_codes[0] for t in tgt_codes)
            code = tgt_codes[0]
            all_codes.add(code)

            self.tgt[code] = VocabEntry.from_corpus(
                tgt_sents, vocab_size, freq_cutoff, is_sentence=True)

        # Keyword and Language indices are source/target independent

        # Create keyword index
        self.keyword = VocabEntry(is_sentence=False)
        for kw in all_keywords:
            self.keyword.add(kw)

        # Create language index
        self.language = VocabEntry(is_sentence=False)
        for code in all_codes:
            self.language.add(code)

    def __repr__(self):
        msg = "Vocab"
        msg += " Source:"
        for key, entry in self.src.items():
            msg += " {}: {} words".format(key, len(entry))

        msg += " Target:"
        for key, entry in self.tgt.items():
            msg += " {}: {} words".format(key, len(entry))

        msg += ", Keyword {} words".format(len(self.keyword))
        msg += ", Language {} words.".format(len(self.language))

        return msg


def read_list(file_path):
    datas = []
    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            corpus_path = line.strip()
            datas.append(corpus_path)
    return datas


if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    print('read in source list: %s' % args['--train-src-list'])
    src_list = read_list(args['--train-src-list'])
    print('read in target list: %s' % args['--train-tgt-list'])
    tgt_list = read_list(args['--train-tgt-list'])

    vocab = Vocab(src_list, tgt_list, int(args['--size']),
                  int(args['--freq-cutoff']))

    print(vocab)
    with open(args['VOCAB_FILE'], 'wb') as fout:
        pickle.dump(vocab, fout)

    print('vocabulary saved to %s' % args['VOCAB_FILE'])
