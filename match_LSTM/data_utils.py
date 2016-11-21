import json
import six
import numpy as np
import random
import os
from nltk.tokenize import RegexpTokenizer
word_tokenize = RegexpTokenizer(r'\w+').tokenize


def format_data(js):
    data_list = js['data']
    fours = []
    for article in data_list:
        for passage in article['paragraphs']:
            context = passage['context'].lower().strip()  # unicode string
            for qa in passage['qas']:
                q = qa['question']  # unicode string
                a = qa['answers']   # list of dicts
                assert len(a) == 1, a
                fours.append((
                    context,
                    q.lower().strip(), 
                    a[0]['text'].lower().strip(), 
                    int(a[0]['answer_start']))) 
    return fours


def create_vocab(fours, cap=None):
    X = [ " ".join(_[:3]) for _ in fours]
    X = "  ".join(X)
    X = word_tokenize(X)
    voca = {"<PAD>": 0, "<UNK>": 1, "<STOP>": 2}

    # calculate frequency
    f = {}
    for t in X:
        if t in f:
            f[t] += 1
        else:
            f[t] = 1

    if cap is not None and cap < len(f):
        print cap
        f = sorted(sorted(six.iteritems(f), key=lambda x: (isinstance(x[0], str), x[0])),
                   key=lambda x: x[1], reverse=True)[:cap-len(voca)]
    else:
        f = f.items()

    # build vocab
    add = len(voca)
    for i, p in enumerate(f):
        voca[p[0]] = i + add
    return voca

def tokenize_data(vocab, data):
    unk_id = vocab['<UNK>']
    # stop_id = vocab['<STOP>']
    w2id = lambda x: vocab.get(x, unk_id)

    result = []
    for index, _ in enumerate(data):
        p, q, a, s = _

        start = len(word_tokenize(p[:s]))
        p = word_tokenize(p)
        a = word_tokenize(a)        
        aid = [start + i for i in range(len(a))]
        if len(aid) == 0 or aid[-1] >= len(p):
            continue
        q = word_tokenize(q)

        p = map(w2id,p)
        q = map(w2id,q)
        # p.append(stop_id)
        # q.append(stop_id)
        # aid.append(len(p)-1)
        result.append([p,q,aid])
    return result

def process_data(js='./squad/train-v1.1.json', save_path='./squad/', cap=None):
    with open(js, 'r') as f:
        squad = json.load(f)
    fours = format_data(squad)
    
    if cap is None:
        vocab_path = os.path.join(save_path, 'vocab_full.js')
    else:
        vocab_path = os.path.join(save_path, 'vocab_%d.js'%cap)

    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    else:
        vocab = create_vocab(fours, cap=cap)
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=4)
    
    ids_path = os.path.join(save_path, 'ids_vocab%d.js'%len(vocab))
    ids = tokenize_data(vocab, fours)
    with open(ids_path, 'w') as f:
        json.dump(ids, f, indent=4)
    return ids

def _transform(d,l,end,pad=0):
    if len(d) >= l:
        d[l-1] = end
        return d[:l], l
    else:
        d.append(end)
        l_ = len(d)
        d = d + [pad]*(l-l_)
        return d, l_

def batchIter(batch_size, data, pLen, qLen, aLen, stop_id=2):
    N = len(data)
    steps = np.ceil(N / float(batch_size))
    steps = int(steps)
    yield steps

    P = np.zeros([batch_size, pLen])
    Q = np.zeros([batch_size, qLen])
    A = np.zeros([batch_size, aLen])

    p_length = np.zeros([batch_size], dtype=np.int)
    q_length = np.zeros([batch_size], dtype=np.int)
    a_length = np.zeros([batch_size], dtype=np.int)

    for idx in range(steps):
        start = idx * batch_size
        end = (idx + 1) * batch_size
        if end > N:
            batch = data[start:]+data[:end-N]
        else:
            batch = data[start:end]

        P.fill(0)
        Q.fill(0)
        A.fill(0)
        p_length.fill(0)
        q_length.fill(0)
        a_length.fill(0)

        for i, sample in enumerate(batch):
            p,q,a = sample
            P[i], p_length[i] = _transform(p, pLen, stop_id)
            Q[i], q_length[i] = _transform(q, qLen, stop_id)
            A[i], a_length[i] = _transform(a, aLen, p_length[i]-1)

        yield idx, P, p_length, Q, q_length, A, a_length

def load_data(data_dir='./squad/', batch_size=64, vocab_size=50000, 
                pLen=300, qLen=20, aLen=10, 
                shuffle=False, split_rate=0.9,
                stop_id=2, data_size=None):
    """
    To take full length: pLen=677, qLen=40, aLen=43, 
    """ 
    ids_path = os.path.join(data_dir, 'ids_vocab%d.js'%vocab_size)
    with open(ids_path, 'r') as f:
        data = json.load(f)
    
    if data_size:
        data = data[:data_size]

    if shuffle:
        data = random.shuffle(data)
    part = int(np.floor(len(data) * split_rate))
    train = data[:part]
    validate = data[part:]

    titer = batchIter(batch_size, train, pLen, qLen, aLen, stop_id=stop_id)
    viter = batchIter(batch_size, validate, pLen, qLen, aLen, stop_id=stop_id)

    tstep = titer.next()
    vstep = viter.next()

    return titer, tstep, viter, vstep


if __name__ == '__main__':
    process_data(cap=50000)