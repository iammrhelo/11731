#!/usr/bin/env python
"""
Modified vocab.py for IWSLT corpus
Generate the vocabulary file for neural network training
A vocabulary file is a mapping of tokens to their indices

Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 20000]
    --freq-cutoff=<int>        frequency cutoff [default: 5]
"""

#python -u gnmt_vocab.py --train-src='../iwslt2017/data/train.all.src' --train-tgt='../iwslt2017/data/train.all.tgt' --size=30000 ../iwslt2017/data/vocab.all.bin

from typing import List
from collections import Counter
from itertools import chain
from docopt import docopt
import pickle
from collections import defaultdict

from utils import read_iwslt_corpus


class VocabEntry(object):
    def __init__(self, code=None, is_sentence=True):
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
    def from_corpus(cls, coded_corpus, size=None, freq_cutoff=5, code=None, is_sentence=True):

        top_words = []

        for key in coded_corpus:
            corpus = coded_corpus[key]
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

            top_words.extend(list(top_k_words))

        top_words = set(top_words)
        print(len(top_words))
        # Create vocab class here
        vocab_entry = cls(code, is_sentence)
        for word in top_words:
            vocab_entry.add(word)
        print(len(vocab_entry))
        
        return vocab_entry


class Vocab(object):
    def __init__(self, src_data, tgt_data, vocab_size, freq_cutoff):
        assert len(src_data) == len(tgt_data)

        src_keywords, src_code, src_sents = list(zip(*src_data))
        tgt_keywords, tgt_code, tgt_sents = list(zip(*tgt_data))

        src_coded_sents = defaultdict(list)
        tgt_coded_sents = defaultdict(list)

        for example in src_data:
            kw, code, sent = example
            src_coded_sents[code].append(sent)

        for example in tgt_data:
            kw, code, sent = example
            tgt_coded_sents[code].append(sent)

        # src_keywords should be equal to target keywords
        assert all(s == t for s, t in zip(src_keywords, tgt_keywords))

        #all_keywords = set(chain(*(src_keywords + tgt_keywords)))
        all_codes = set(src_code + tgt_code)

        print('initialize source vocabulary ..')
        self.src = VocabEntry.from_corpus(src_coded_sents, vocab_size, freq_cutoff)

        print('initialize target vocabulary ..')
        self.tgt = VocabEntry.from_corpus(tgt_coded_sents, vocab_size, freq_cutoff)

        # Add keywords to both source and target
        #for kw in all_keywords:
        #    self.src.add(kw)
        #    self.tgt.add(kw)
        print(all_codes)

        # Add language codes to both source and target
        for code in all_codes:
            self.src.add(code)

    def __repr__(self):
        return 'Vocab(source {} words, target {} words)'\
            .format(len(self.src), len(self.tgt))


if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    print('read in source sentences: %s' % args['--train-src'])
    src_data = read_iwslt_corpus(args['--train-src'], source='src')
    print('total number of sentences in source %d', len(src_data))

    print('read in target sentences: %s' % args['--train-tgt'])
    tgt_data = read_iwslt_corpus(args['--train-tgt'], source='tgt')
    print('total number of sentences in target %d', len(tgt_data))

    vocab = Vocab(src_data, tgt_data, int(
        args['--size']), int(args['--freq-cutoff']))
    print(vocab)

    with open(args['VOCAB_FILE'], 'wb') as fout:
        pickle.dump(vocab, fout)

    print('vocabulary saved to %s' % args['VOCAB_FILE'])
