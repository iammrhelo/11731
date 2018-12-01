# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    hypernmt.py train [options]
    hypernmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    hypernmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    hypernmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file [default: ./data/dev_source.txt]
    --train-tgt=<file>                      train target file [default: ./data/dev_target.txt]
    --dev-src=<file>                        dev source file [default: ./data/dev_source.txt]
    --dev-tgt=<file>                        dev target file [default: ./data/dev_target.txt]
    --vocab=<file>                          vocab file [default: ./data/vocab.all.bin]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --num-layers=<int>                      number of layers for encoder and decoder [default: 1]
    --bidirectional                         use bidirectional for encoder
    --attn-type=<str>                       type of attention to use [default: Concat]
    --mask-attn=<bool>                      mask src encodings [default: False]
    --domain-embed-size=<int>               domain embedding size [default: 2]
    --relevant-information-size=<int>       relevant information size [default: 5]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how mtyping.Any iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how mtyping.Any trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --optimizer=<str>                       optimizer [default: Adam]
    --lr=<float>                            learning rate [default: 0.001]
    --label-smoothing=<float>               label smoothing parameter [default: 0.1]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: work_dir.debug]
    --valid-niter=<int>                     perform validation after how mtyping.Any iterations [default: 10]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --length-norm-alpha=<float>             beam search length normalization alpha parameter [default: 0.2]
    --length-norm-beta=<float>              beam search length normalization beta parameter [default: 0.2]
    --model-path=<file>                     path to model file
    --lang1=<str>                           add descriptions
    --lang2=<str>                           add descriptions
    --lang-embed-size=<int>                 add descriptions
    --langsrcs=<str>                        add descriptions
    --langtgts=<str>                        add descriptions
"""

import gc
import math
import os
import pickle
import sys
import time
import typing
from collections import namedtuple
from copy import deepcopy
from typing import Dict, List, Set, Tuple, Union

import numpy as np
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from docopt import docopt
from nltk.translate.bleu_score import (SmoothingFunction, corpus_bleu,
                                       sentence_bleu)
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from misc import LabelSmoothing
from models import GlobalAttention, LuongDecoder, HyperEncoder
from utils import (batch_iter, hyper_batch_iter, hyper_read_corpus,
                   input_transpose, read_corpus, hyper_zeroshot_batch_iter)
from vocab import Vocab, VocabEntry

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class HyperNMT(nn.Module):
    """
    A basic implementation of a Neural Machine Translation model
    """

    def __init__(self, opt):
        super(HyperNMT, self).__init__()
        # LSTM
        self.embed_size = opt["embed_size"]
        self.hidden_size = opt["hidden_size"]
        self.num_layers = opt["num_layers"]
        self.bidirectional = opt["bidirectional"]
        self.attn_type = opt["attn_type"]
        self.mask_attn = opt["mask_attn"]
        self.dropout_rate = opt["dropout_rate"]

        # HYPER
        self.lang_embedding_size = opt["lang_embed_size"]
        self.num_languages = opt["num_languages"]
        self.enc_lang_embedding = torch.nn.Embedding(self.num_languages, self.lang_embedding_size)
        self.dec_lang_embedding = torch.nn.Embedding(self.num_languages, self.lang_embedding_size)

        # XXXself.domain_embed_size = opt["domain_embed_size"]
        # XXXself.relevant_information_size = opt["relevant_information_size"]

        self.vocab = opt["vocab"]
        self.label_smoothing = opt["label_smoothing"]
        self.use_cuda = opt["use_cuda"]

        # Build Encoder, Decoder, and Attention and CPG
        # Encoder
        encoder_opt = deepcopy(opt)
        encoder_opt["num_embeddings"] = len(self.vocab.src)
        self.hyper_encoder = HyperEncoder(encoder_opt)

        encoder_hidden_size = (int(self.bidirectional)+1) * self.hidden_size
        decoder_hidden_size = self.hidden_size

        # Attention
        attn = GlobalAttention(
            self.attn_type, self.mask_attn, encoder_hidden_size, decoder_hidden_size)

        # Decoder
        decoder_opt = deepcopy(opt)
        decoder_opt["num_embeddings"] = len(self.vocab.tgt)
        #TODO HyperLuongDecoder
        self.decoder = LuongDecoder(decoder_opt, attn)

        # CPG
        # XXXcpg_opt = deepcopy(opt)
        # XXXcpg_opt["encoder_rnn_param_size"] = self.encoder.get_rnn_parameter_size()
        # XXXcpg_opt["decoder_rnn_param_size"] = self.decoder.get_rnn_parameter_size()
        # XXXself.cpg = CPG(cpg_opt)

        # Evaluation
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab.tgt.pad_id, reduction="none")
        # self.criterion = LabelSmoothing(
        #     size=len(self.vocab.tgt.size), padding_idx=self.vocab.tgt.pad_id, smoothing=self.label_smoothing)

        if self.use_cuda:
            self.cuda()

    def init_weights(self, uniform_weight=0.1):
        """
        Initialize weights for all modules
        """
        for param in self.parameters():
            init.uniform_(param, -uniform_weight, uniform_weight)

    def __call__(self, src_sents: List[List[str]], tgt_sents: List[List[str]], domain: int) -> Tensor:
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`
            domain: domain index integer
        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """

        # Contextual Parameter Generation
        domain_tensor = torch.LongTensor([domain])
        if self.use_cuda:
            domain_tensor = domain_tensor.cuda()
        theta_enc, theta_dec = self.cpg.forward(domain_tensor)
        self.encoder.set_rnn_parameters(theta_enc)
        self.decoder.set_rnn_parameters(theta_dec)

        # Encode
        src_encodings, src_lengths, decoder_init_state = self.encode(src_sents)

        # Decode
        scores = self.decode(
            src_encodings, src_lengths, decoder_init_state, tgt_sents)
        return scores

    def sents2tensor(self, sents: List[List[str]], vocab: typing.Any) -> Tensor:
        """
        Takes a mini-batch of sentences, convert into LongTensor
        Args:
            sents : list of sentence tokens
        Returns:
            sents_tensor: padded sents tensor with shape(length, batch_size)
        """
        sent_ids = [vocab.words2indices(sent) for sent in sents]
        transposed_ids = input_transpose(sent_ids, vocab.pad_id)
        sents_tensor = torch.LongTensor(transposed_ids)
        if self.use_cuda:
            sents_tensor = sents_tensor.cuda()
        return sents_tensor

    def encode(self, src_sents: List[List[str]]) -> Tuple[Tensor, typing.Any]:
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens, already sorted in decreasing order by length

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable
                with shape (batch_size, source_sentence_length, encoding_dim), or in other formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """
        # Convert words to tensor
        src_tensor = self.sents2tensor(src_sents, self.vocab.src)
        src_lengths = (src_tensor != self.vocab.src.pad_id).sum(dim=0)

        # (length, batch_size, dim)
        encoder_output, encoder_hidden = self.encoder(src_tensor, src_lengths)
        return encoder_output, src_lengths, encoder_hidden

    def decode(self, src_encodings: Tensor, src_lengths: Tensor, decoder_init_state: typing.Any, tgt_sents: List[List[str]] = None) -> Tensor:
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """
        # tgt_sents 2 tensor
        # (length, batch_size)
        tgt_tensor = self.sents2tensor(tgt_sents, self.vocab.tgt)
        # Here we feed in the target output for log-likelihood prediction
        # (length, batch_size, classes)
        decoder_hidden = decoder_init_state
        decoder_input = tgt_tensor[:-1]
        decoder_true = tgt_tensor[1:]

        decoder_pred, _ = self.decoder.forward(
            decoder_input, decoder_hidden, src_encodings, src_lengths, decoder_true)

        # Permute for loss calculation
        # (batch_size, classes, length)
        decoder_pred = decoder_pred.permute(1, 2, 0)
        # (batch_size, length)
        decoder_true = decoder_true.permute(1, 0)

        # (batch_size, length)
        loss = self.criterion(decoder_pred, decoder_true)

        scores = loss.sum(dim=1)  # sum over length
        return scores

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70, alpha:float=0.2, beta:float=0.2) -> List[Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN
            alpha: length normalization factor
            beta: length normaliziation factor

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        start_id = self.vocab.tgt["<s>"]
        stop_id = self.vocab.tgt["</s>"]

        with torch.no_grad():
            # (length, 1)
            src_tensor = self.sents2tensor([src_sent], self.vocab.src)
            src_lengths = (src_tensor != self.vocab.src.pad_id).sum(dim=0)

            # Get information from encoder
            encoder_output, decoder_init_state = self.encoder.forward(
                src_tensor, src_lengths)

            # Do debugging here
            # src_sent: [ a, b, c, d ]
            # src_tensor: [ 1, 4, 6, 7, 5 ]
            # ground_truth: [ <s>, b, c, d, a, </s> ]
            # tgt_tensor:   [   1, 4, 6, 7, 5,    2 ]

            # beam_states: list of tuples of (indices, score, decoder_hidden)
            init_state = ([start_id], 0.0, decoder_init_state)
            beam_states = [init_state]
            for step in range(max_decoding_time_step):
                next_beam_states = []
                for prev_indices, prev_score, prev_decoder_hidden in beam_states:
                    prev_word_id = prev_indices[-1]

                    if prev_word_id == stop_id:  # </s>
                        next_beam_states.append(
                            (prev_indices, prev_score, prev_decoder_hidden))
                        continue

                    # Initialize decoder_input (1, 1)
                    decoder_input = torch.LongTensor([[prev_word_id]])
                    if self.use_cuda:
                        decoder_input = decoder_input.cuda()

                    decoder_output, decoder_hidden = self.decoder.forward(
                        decoder_input, prev_decoder_hidden, encoder_output)

                    # Since we have only 1 element, squeeze to 1 dim
                    decoder_output = decoder_output.squeeze()
                    decoder_probs = F.softmax(decoder_output, dim=0)

                    # Apply length normalization here
                    scores = decoder_probs.log()  # log score for addition

                    # Top beam_size candidates and move to numpy
                    cand_scores, cand_indices = scores.topk(beam_size)
                    if self.use_cuda:
                        cand_scores = cand_scores.cpu()
                        cand_indices = cand_indices.cpu()

                    cand_scores = cand_scores.numpy()
                    cand_indices = cand_indices.numpy()

                    # Add possible candidates to next_beam_states
                    for word_score, word_id in zip(cand_scores, cand_indices):
                        candidate = (prev_indices + [word_id],
                                     prev_score + word_score, decoder_hidden)
                        next_beam_states.append(candidate)

                # Choose among the ones with highest scores
                next_beam_states = sorted(
                    next_beam_states, key=lambda c: c[1], reverse=True)[:beam_size]

                beam_states = next_beam_states

                # Breaks if all beam search candidates have ended
                if sum(c[0][-1] == stop_id for c in beam_states) == beam_size:
                    break

        # Convert beam_states into hypothesis
        hypotheses = []
        for word_indices, log_score, _ in beam_states:
            sent = self.vocab.tgt.indices2words(
                word_indices[1:-1])
            hyp = Hypothesis(value=sent, score=log_score)
            hypotheses.append(hyp)
        return hypotheses

    def hyper_evaluate_ppl(self, dev_data: List[typing.Any], batch_size: int = 16):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of domain data, which are list of sentences
            batch_size: batch size

        Returns:
            ppl: the perplexity on dev sentences
        """
        # Evaluation mode
        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`
        with torch.no_grad():
            for domain_idx, domain_dev_data in enumerate(dev_data):
                for src_sents, tgt_sents in batch_iter(domain_dev_data, batch_size):
                    loss = self.__call__(
                        src_sents, tgt_sents, domain_idx).sum()
                    cum_loss += loss
                    # omitting the leading `<s>`
                    tgt_word_num_to_predict = sum(
                        len(s[1:]) for s in tgt_sents)
                    cum_tgt_words += tgt_word_num_to_predict

            ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

    @staticmethod
    def load(model_path: str, use_cuda=False):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        if use_cuda:
            model = torch.load(model_path)
        else:
            model = torch.load(
                model_path, map_location=lambda storage, loc: storage)
            model.use_cuda = False
        model.encoder.rnn.flatten_parameters()
        model.decoder.rnn.flatten_parameters()
        return model

    def save(self, path: str):
        """
        Save current model to file
        """
        torch.save(self, path)


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score

def get_data_old(srcs, tgts):
    all_languages = []

    train_srcs = []
    train_tgts = []
    train_srcs_langs = []
    train_tgts_langs = []

    dev_srcs = []
    dev_tgts = []
    dev_srcs_langs = []
    dev_tgts_langs = []

    list_src = srcs.split('-')
    list_tgt = tgts.split('-')
    for src,tgt in zip(list_src, list_tgt):
        all_languages.append(src)
        all_languages.append(tgt)

    lang_map = build_lang_map(all_languages)

    train_data = []
    dev_data = []

    for src,tgt in zip(list_src, list_tgt):
        print(src,tgt)
        train_src  =f"data/train.en-{src}.{src}.txt"
        train_tgt  =f"data/train.en-{src}.en.txt"
        dev_src =f"data/dev.en-{src}.{src}.txt"
        dev_tgt =f"data/dev.en-{src}.en.txt"
        train_data_src = read_corpus(train_src, source='src')
        train_data_tgt = read_corpus(train_tgt, source='tgt')

        train_src_lang = [lang_map[src] for i in range(len(train_data_src))]
        train_tgt_lang = [lang_map[tgt] for i in range(len(train_data_tgt))]

        dev_data_src = read_corpus(dev_src, source='src')
        dev_data_tgt = read_corpus(dev_tgt, source='tgt')

        dev_src_lang = [lang_map[src] for i in range(len(dev_data_src))]
        dev_tgt_lang = [lang_map[tgt] for i in range(len(dev_data_tgt))]

        train_data.append([train_data_src, train_data_tgt, train_src_lang, train_tgt_lang])
        dev_data.append([dev_data_src, dev_data_tgt, dev_src_lang, dev_tgt_lang])
        # train_srcs.append(train_data_src)
        # train_tgts.append(train_data_tgt)
        # train_srcs_langs.append(train_src_lang)
        # train_tgts_langs.append(train_tgt_lang)

        # dev_srcs.append(dev_data_src)
        # dev_tgts.append(dev_data_tgt)
        # dev_srcs_langs.append(dev_src_lang)
        # dev_tgts_langs.append(dev_tgt_lang)
    return train_data, dev_data, lang_map
    # return train_srcs, train_tgts, train_srcs_langs, train_tgts_langs, dev_srcs, dev_tgts, dev_srcs_langs, dev_tgts_langs, lang_map


def build_lang_map(all_languages):
    lang_map = {}
    for lang in all_languages:
        lang_map[lang] = len(lang_map)
    return lang_map

def train(args: Dict[str, str]):
    train_data, dev_data, language_map = get_data_old(args['--langsrcs'], args['--langtgts'])
    num_languages = len(language_map)


    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    optimizer = args['--optimizer']
    lr = float(args['--lr'])
    uniform_init = float(args["--uniform-init"])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    work_dir = args['--save-to']
    model_save_path = os.path.join(work_dir, 'model.bin')
    optim_save_path = os.path.join(work_dir, 'optim.bin')

    vocab = pickle.load(open(args['--vocab'], 'rb'))
    print('src vocab', len(vocab.src))
    print('tgt vocab', len(vocab.tgt))


    model_opt = {
        "embed_size": int(args['--embed-size']),
        "hidden_size": int(args['--hidden-size']),
        "num_layers": int(args['--num-layers']),
        "dropout_rate": float(args['--dropout']),
        "bidirectional": bool(args['--bidirectional']),
        "attn_type": args['--attn-type'],
        "mask_attn": args['--mask-attn'] == "True",
        "vocab": vocab,
        "label_smoothing": float(args['--label-smoothing']),
        "use_cuda": bool(args["--cuda"]),
        'lang_embed_size': int(args['--lang-embed-size']),
        "num_languages": num_languages
    }
    if not torch.cuda.is_available():
        model_opt["use_cuda"] = False

    model = HyperNMT(model_opt)
    model.init_weights(uniform_init)

    if args["--model-path"]:
        model = HyperNMT.load(args["--model-path"])

    if optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Unknown optimizer: {}".format(optimizer))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print("begin Maximum Likelihood training")


    for idx, (src_sents, tgt_sents, src_langs, tgt_langs) in hyper_zeroshot_batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
        print(idx)
        pass
        foo
        low_batches = round(len(train_low_data)/train_batch_size)
        high_batches = round(len(train_high_data)/train_batch_size)
        print("Low and High", low_batches+high_batches)

        batch_order = [0]*low_batches + [1]*high_batches
        np.random.shuffle(batch_order)

        train_low_batches = hyper_batch_iter(train_low_data, batch_size=train_batch_size, shuffle=True)
        train_high_batches = hyper_batch_iter(train_high_data, batch_size=train_batch_size, shuffle=True)
        for batch in batch_order:
            if batch == 0:
                src_sents, tgt_sents, src_langs = next(train_low_batches)
                #low
            elif batch == 1:
                #high
                src_sents, tgt_sents, src_langs = next(train_high_batches)
    foo
    while True:
        epoch += 1

        for src_sents, tgt_sents, src_langs in hyper_batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            model.train()
            train_iter += 1

            batch_size = len(src_sents)

            optimizer.zero_grad()
            # (batch_size)
            scores = model(src_sents, tgt_sents, src_langs)
            loss = scores.sum()

            # Optimizer here
            loss.backward()
            clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            if model.use_cuda:
                report_loss += loss.data.cpu().numpy()
                cum_loss += loss.data.cpu().numpy()
            else:
                report_loss += loss.data.numpy()
                cum_loss += loss.data.numpy()

            tgt_words_num_to_predict = sum(
                len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict

            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f '
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(
                                                                                             report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words /
                                                                                         (time.time(
                                                                                         ) - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)
                sys.stdout.flush()
                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                             cum_loss / cumulative_examples,
                                                                                             np.exp(
                                                                                                 cum_loss / cumulative_tgt_words),
                                                                                             cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)
                model.eval()
                # compute dev. ppl and bleu
                # dev batch size can be a bit larger
                dev_ppl = model.evaluate_ppl(dev_low_data, batch_size=32)
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' %
                      (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(
                    hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print(
                        'save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # You may also save the optimizer's state
                    torch.save(optimizer, optim_save_path)
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        print(
                            'load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        model = HyperNMT.load(model_save_path)
                        print('restore parameters of the optimizers',
                              file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before
                        optimizer = torch.load(optim_save_path)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)

                gc.collect()

    # Never ending sampling
    # for domain_idx, (src_sents, tgt_sents) in hyper_batch_iter(train_data, batch_size=train_batch_size, shuffle=True):

    #     model.train()
    #     train_iter += 1

    #     batch_size = len(src_sents)

    #     optimizer.zero_grad()
    #     # (batch_size)
    #     scores = model(src_sents, tgt_sents, domain_idx)
    #     loss = scores.sum()

    #     # Optimizer here
    #     loss.backward()
    #     clip_grad_norm_(model.parameters(), clip_grad)
    #     optimizer.step()

    #     if model.use_cuda:
    #         report_loss += loss.data.cpu().numpy()
    #         cum_loss += loss.data.cpu().numpy()
    #     else:
    #         report_loss += loss.data.numpy()
    #         cum_loss += loss.data.numpy()

    #     tgt_words_num_to_predict = sum(
    #         len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
    #     report_tgt_words += tgt_words_num_to_predict
    #     cumulative_tgt_words += tgt_words_num_to_predict

    #     report_examples += batch_size
    #     cumulative_examples += batch_size

    #     if train_iter % log_every == 0:
    #         print('iter %d, avg. loss %.2f, avg. ppl %.2f '
    #               'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (train_iter,
    #                                                                                  report_loss / report_examples,
    #                                                                                  math.exp(
    #                                                                                      report_loss / report_tgt_words),
    #                                                                                  cumulative_examples,
    #                                                                                  report_tgt_words /
    #                                                                                  (time.time(
    #                                                                                  ) - train_time),
    #                                                                                  time.time() - begin_time), file=sys.stderr)

    #         train_time = time.time()
    #         report_loss = report_tgt_words = report_examples = 0.

    #     # the following code performs validation on dev set, and controls the learning schedule
    #     # if the dev score is better than the last check point, then the current model is saved.
    #     # otherwise, we allow for that performance degeneration for up to `--patience` times;
    #     # if the dev score does not increase after `--patience` iterations, we reload the previously
    #     # saved best model (and the state of the optimizer), halve the learning rate and continue
    #     # training. This repeats for up to `--max-num-trial` times.
    #     if train_iter % valid_niter == 0:
    #         print('iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (train_iter,
    #                                                                            cum_loss / cumulative_examples,
    #                                                                            np.exp(
    #                                                                                cum_loss / cumulative_tgt_words),
    #                                                                            cumulative_examples), file=sys.stderr)

    #         cum_loss = cumulative_examples = cumulative_tgt_words = 0.
    #         valid_num += 1

    #         print('begin validation ...', file=sys.stderr)
    #         model.eval()
    #         # compute dev. ppl and bleu
    #         # dev batch size can be a bit larger

    #         dev_ppl = model.hyper_evaluate_ppl(dev_data, batch_size=32)

    #         valid_metric = -dev_ppl

    #         print('validation: iter %d, dev. ppl %f' %
    #               (train_iter, dev_ppl), file=sys.stderr)

    #         is_better = len(hist_valid_scores) == 0 or valid_metric > max(
    #             hist_valid_scores)
    #         hist_valid_scores.append(valid_metric)

    #         if is_better:
    #             patience = 0
    #             print(
    #                 'save currently the best model to [%s]' % model_save_path, file=sys.stderr)
    #             model.save(model_save_path)

    #             # You may also save the optimizer's state
    #             torch.save(optimizer, optim_save_path)
    #         elif patience < int(args['--patience']):
    #             patience += 1
    #             print('hit patience %d' % patience, file=sys.stderr)

    #             if patience == int(args['--patience']):
    #                 num_trial += 1
    #                 print('hit #%d trial' % num_trial, file=sys.stderr)
    #                 if num_trial == int(args['--max-num-trial']):
    #                     print('early stop!', file=sys.stderr)
    #                     exit(0)

    #                 # decay learning rate, and restore from previously best checkpoint
    #                 lr = lr * float(args['--lr-decay'])
    #                 print(
    #                     'load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

    #                 # load model`
    #                 model = HyperNMT.load(
    #                     model_save_path, use_cuda=bool(args["--cuda"]))
    #                 print('restore parameters of the optimizers',
    #                       file=sys.stderr)

    #                 # You may also need to load the state of the optimizer saved before
    #                 optimizer = torch.load(optim_save_path)
    #                 for param_group in optimizer.param_groups:
    #                     param_group['lr'] = lr

    #                 # reset patience
    #                 patience = 0

    #         """
    #         if epoch == int(args['--max-epoch']):
    #             print('reached maximum number of epochs!', file=sys.stderr)
    #             exit(0)
    #         """
    #         gc.collect()


def beam_search(model: HyperNMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int, alpha: float, beta: float) -> List[List[Hypothesis]]:
    was_training = model.training

    hypotheses = []
    try:
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(
                src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step, alpha=alpha, beta=beta)
            hypotheses.append(example_hyps)
    except KeyboardInterrupt:
        print("Keyboard interrupted!")

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    use_cuda = bool(args['--cuda'])
    model = HyperNMT.load(args['MODEL_PATH'], use_cuda)
    model.eval()
    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(
                                 args['--max-decoding-time-step']),
                             alpha=float(args['--length-norm-alpha']),
                             beta=float(args["--length-norm-beta"]))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(
            test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)
    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
