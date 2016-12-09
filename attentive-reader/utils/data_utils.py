# Modification of https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/rnn/translate/data_utils.py
#
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import re
import codecs
import json
import time
from tqdm import *
from glob import glob
# from nltk.tokenize import RegexpTokenizer
import numpy as np
import random
from tensorflow.python.platform import gfile

from nltk import TreebankWordTokenizer
from string import punctuation
_tokenrize = TreebankWordTokenizer().tokenize

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")
# word_tokenize = RegexpTokenizer(r'\w+').tokenize

PAD_ID = 0
UNK_ID = 1
STOP_ID = 2


def create_vocab(doc_path, cap=None, save_full_to=None, normalize_digits=False):
    start = time.time()
    fp = codecs.open(doc_path, mode='r')
    f = {}
    for i, line in enumerate(tqdm(fp)):
        line = line.lower()
        if normalize_digits:
            line = re.sub(_DIGIT_RE, "0", line)
        X = token(line)

        for t in X:
            if t in f:
                f[t] += 1
            else:
                f[t] = 1

    print('Calculate Frequency done %4.4f' % (time.time() - start))
    return f

    if save_full_to:
        with open(save_full_to, 'a+') as save:
            json.dump(f, save, ensure_ascii=False)

    start = time.time()
    if cap is not None and cap < len(f):
        print(cap)
        f = sorted(sorted(six.iteritems(f), key=lambda x: (isinstance(x[0], str), x[0])),
                   key=lambda x: x[1], reverse=True)[:cap]
    else:
        f = f.items()
    print('Cut done %4.4f' % (time.time() - start))

    # build vocab
    start = time.time()
    voca = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, "<STOP>": STOP_ID}
    add = len(voca)
    for i, p in enumerate(f):
        voca[p[0]] = i + add
    print('Vocab Created %4.4f' % (time.time() - start))
    return voca


def token(words):
    ts = _tokenrize(words)
    ts = [x.strip(punctuation) for x in ts]
    ts = [x for x in ts if len(x) > 0]
    return ts


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=token, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: a string, the sentence to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """
    words = tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocab,
                      tokenizer=token, normalize_digits=False, save=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    # if not gfile.Exists(target_path):
    if True:
        with open(data_path, "r") as data_file:
            counter = 0
            results = []
            for line in data_file:
                if counter == 0:
                    results.append(line)
                elif counter == 4:
                    entity, ans = line.split(":", 1)
                    try:
                        results.append("%s:%s" % (vocab[entity[:]], ans))
                    except:
                        continue
                else:
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                      normalize_digits)
                    results.append(" ".join([str(tok)
                                             for tok in token_ids]) + "\n")
                if line == "\n":
                    counter += 1

            try:
                len_d, len_q = len(results[2].split()), len(results[4].split())
            except:
                return
            if save:
                with open("%s_%s" % (target_path, len_d + len_q), "w") as tokens_file:
                    tokens_file.writelines(results)
            return results
        # assert data_file.closed and tokens_file.closed, (data_file.closed , tokens_file.closed)


def get_all_context(dir_name, context_fname):
    context = ""
    for fname in tqdm(glob(os.path.join(dir_name, "*.question"))):
        with open(fname) as f:
            try:
                lines = f.read().split("\n\n")
                context += lines[1] + "\n "
                context += lines[4].replace(":", " ") + " "
            except:
                print(" [!] Error occured for %s" % fname)
    print(" [*] Writing %s ..." % context_fname)
    with open(context_fname, 'wb') as f:
        f.write(context)
    return context


def questions_to_token_ids(data_path, vocab_fname):
    import pickle as pk
    with open(vocab_fname, 'r') as f:
        vocab = pk.load(f)
    vocab_size = len(vocab)

    for fname in tqdm(glob(os.path.join(data_path, "*.question"))):
        data_to_token_ids(fname, fname + ".ids%s" % vocab_size, vocab)


def prepare_data(data_dir, dataset_name, vocab_size):
    train_path = os.path.join(data_dir, dataset_name, 'questions', 'training')

    context_fname = os.path.join(
        data_dir, dataset_name, '%s.context' % dataset_name)
    vocab_fname = os.path.join(
        data_dir, dataset_name, '%s.vocab%s' % (dataset_name, vocab_size))

    if not os.path.exists(context_fname):
        print(" [*] Combining all contexts for %s in %s ..." %
              (dataset_name, train_path))
        context = get_all_context(train_path, context_fname)
    else:
        context = gfile.GFile(context_fname, mode="r").read()
        print(" [*] Skip combining all contexts")

    if not os.path.exists(vocab_fname):
        print(" [*] Create vocab from %s to %s ..." %
              (context_fname, vocab_fname))
        create_vocabulary(vocab_fname, context, vocab_size)
    else:
        print(" [*] Skip creating vocab")

    print(" [*] Convert data in %s into vocab indicies..." % (train_path))
    questions_to_token_ids(train_path, vocab_fname)


def load_vocab(data_dir, dataset_name, vocab_size):
    vocab_fname = os.path.join(
        data_dir, dataset_name, "%s.vocab%s" % (dataset_name, vocab_size))
    print(" [*] Loading vocab from %s ..." % vocab_fname)
    return initialize_vocabulary(vocab_fname)


def data_iter(flist, max_nstep, max_query_step, batch_size=None, vocab_size=264588, shuffle_data=True):
    if batch_size is None:
        batch_size = len(flist)

    steps = np.ceil(len(flist) / float(batch_size))
    steps = int(steps)
    yield steps

    y = np.zeros([batch_size, vocab_size])
    ds = np.zeros([batch_size, max_nstep])
    qs = np.zeros([batch_size, max_query_step])

    d_length = np.zeros([batch_size], dtype=np.int)
    q_length = np.zeros([batch_size], dtype=np.int)

    for s in range(steps):
        head = s * batch_size
        end = (s + 1) * batch_size
        if end <= len(flist):
            files = flist[head:end]
        else:
            files = flist[head:] + flist[:end - len(flist)]

        if shuffle_data:
            random.shuffle(files)

        y.fill(0)
        ds.fill(0)
        qs.fill(0)

        for idx, fname in enumerate(files):
            with open(fname) as f:
                _, document, question, answer, _ = f.read().split("\n\n")

            document = [int(d) for d in document.split()]
            question = [int(q) for q in question.split()]

            if len(document) > max_nstep:
                ds[idx] = document[:max_nstep]
                d_length[idx] = max_nstep
            else:
                ds[idx][:len(document)] = document
                d_length[idx] = len(document)

            if len(question) > max_query_step:
                qs[idx] = question[:max_query_step]
                q_length[idx] = max_query_step
            else:
                qs[idx][:len(question)] = question
                q_length[idx] = len(question)

            y[idx][int(answer)] = 1

        yield s, ds, d_length, qs, q_length, y


def fetch_files(data_dir, dataset_name, vocab_size):
    train = glob(os.path.join(data_dir, dataset_name, "questions",
                              "training", "*.question.ids%s_*" % (vocab_size)))
    validate = glob(os.path.join(data_dir, dataset_name, "questions",
                                 "validation", "*.question.ids%s_*" % (vocab_size)))
    return train, validate


def load_dataset(data_dir, dataset_name, vocab_size, batch_size, max_nstep, max_query_step, split_rate=0.9, size=None, shuffle_data=True):

    traFl, valFl = fetch_files(data_dir, dataset_name, vocab_size)
    if size is not None:
        traFl = traFl[:size]

    if shuffle_data:
        random.shuffle(traFl)
        random.shuffle(valFl)

    titer = data_iter(traFl, max_nstep, max_query_step,
                      batch_size, vocab_size=vocab_size, shuffle_data=False)
    viter = data_iter(valFl, max_nstep, max_query_step,
                      batch_size, vocab_size=vocab_size, shuffle_data=False)

    tstep = titer.next()
    vstep = viter.next()

    return titer, tstep, viter, vstep


if __name__ == '__main__':
    pass
