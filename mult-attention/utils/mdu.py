import json
import numpy as np
import random
import os
from tqdm import tqdm
import re
from glob import glob

_START_VOCAB = ["<PAD>", "<UNK>", "<STOP>"]


def format_data(js):
    data_list = js['data']
    fours = []
    for article in data_list:
        for passage in article['paragraphs']:
            context = passage['context'].strip()  # unicode string
            for qa in passage['qas']:
                q = qa['question']  # unicode string
                a = qa['answers']   # list of dicts
                # assert len(a) == 1, a
                if len(a) > 1:
                    print 'multiple answer', a
                    a = a[0:1]
                fours.append((
                    context,
                    q.strip(),
                    a[0]['text'].strip(),
                    int(a[0]['answer_start'])))
    return fours

def filter_data(fours, exps=u"([^A-Z]{2,6})([.?!;]+)(\s+[A-Z]\w*|$)"):
    good = []
    abr = []
    bad = []
    mys = []
    for c, q, a, start in tqdm(fours):
        into = False
        # get delimiter location
        period_loc = [_.start(2) + 1 for _ in re.finditer(exps, c)]
        period_loc.insert(0, 0)
        if period_loc[-1] != len(c):
            period_loc.append(len(c))

        # get answer location
        for i in range(len(period_loc) - 1):
            h = period_loc[i]
            t = period_loc[i + 1]
            if h <= start and start < t:
                into = True

                if c[start] != a[0]:
                    bad.append([c, q, a, start, period_loc, i])
                elif start + len(a) > t:
                    abr.append([c, q, a, start, period_loc, i])
                else:
                    good.append([c, q, a, start, period_loc, i])

                break

        if not into:
            mys.append([c, q, a, start, period_loc, None])

    print "#good:%d #bad_sample:%d #confusing abbrevation:%d #mysterious:%d #all:%d" \
        % (len(good), len(bad), len(abr), len(mys), len(fours))
    return good, bad, abr, mys


def token_sample(data, replace=u"['\",\/#$%\^&\*:{}=\-_`~()\[\]\s]+", normalize_digit=True):
    c, q, a, start, period_loc, asi = data
    sentence = []
    for i in range(len(period_loc) - 1):
        s = c[period_loc[i]:period_loc[i + 1]].strip('. ').lower()
        s = re.sub(replace, ' ', s)
        if normalize_digit:
            s = re.sub('\d', '0', s)
        s = s.strip(' ').split(' ')
        sentence.append(s)

    q = q.strip(' ?').lower()
    q = re.sub(replace, ' ', q)
    if normalize_digit:
        q = re.sub('\d', '0', q)
    q = q.strip(' ').split(' ')

    return [sentence, q, asi]


def create_vocab(data_triple, cap=None):
    X = []
    for sentence, q, asi in data_triple:
        sentence.append(q)
        for s in sentence:
            X.extend(s)

    # calculate frequency
    f = {}
    for t in X:
        if t in f:
            f[t] += 1
        else:
            f[t] = 1

    vocab_list = _START_VOCAB + sorted(f, key=f.get, reverse=True)
    if cap is not None and cap < len(f):
        vocab_list = vocab_list[:cap]

    # build vocab
    vocab = {k: i for i, k in enumerate(vocab_list)}
    return vocab

def data2id(data, vocab, unk="<UNK>"):
    unk_id = vocab[unk]
    trans = lambda x: vocab.get(x,unk_id)
    ids = []
    for sen, qa, aid in data:
        sen = [ map(trans, _) for _ in sen ]
        qa = map(trans, qa)
        ids.append([sen,qa,aid])
    return ids

def token_data(origin_file, save_name, normalize_digit=True):
    if os.path.exists(save_name):
        with open(save_name, 'r') as f:
            formated = json.load(f)
    else:
        with open(origin_file, 'r') as f:
            squad = json.load(f)
        fours = format_data(squad)
        good, bad, abr, mys = filter_data(fours)
        formated = [ token_sample(_, normalize_digit=normalize_digit) for _ in good ]
        with open(save_name, 'w') as f:
            json.dump(formated, f, indent=4)
    return formated

def process_data(data_path='../data/squad', save_path='../data/squad/', cap=None, normalize_digit=True):
    
    train_js = glob( os.path.join(data_path, 'train-v*.json') )[0]
    dev_js = glob( os.path.join(data_path, 'dev-v*.json') )[0]

    # token data
    formated_save_path = os.path.join(save_path, 'formated')
    train_token = token_data(train_js, formated_save_path+'_train.js', normalize_digit=normalize_digit)
    dev_token = token_data(dev_js, formated_save_path+'_dev.js', normalize_digit=normalize_digit)

    # load vocab
    if cap is None:
        vocab_path = os.path.join(save_path, 'vocab_full.js')
    else:
        vocab_path = os.path.join(save_path, 'vocab_%d.js' % cap)
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        print 'Vocabulary loaded'
    else:
        vocab = create_vocab(formated, cap=cap)
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=4)
        print 'Vocabulary created'

    # transform data
    ids_path = os.path.join(save_path, 'ids_vocab%d' % len(vocab))
    train_ids = data2id(train_token, vocab)
    dev_ids = data2id(dev_token, vocab)

    def save(_fname, _data):
        with open(_fname, 'w') as f:
            for seq, qur, aid in _data:
                N = len(seq)
                f.write( "%d\n"%N )
                for s in seq:
                    f.write( " ".join(map(str,s))+'\n' )
                f.write( " ".join(map(str,qur))+'\n' )
                f.write( "%d\n\n"%aid )

    save( ids_path+'_train.txt', train_ids )
    save( ids_path+'_dev.txt',   dev_ids )

def _transform(d, l, end, pad=0):
    if len(d) >= l:
        d[l - 1] = end
        return d[:l], l
    else:
        d.append(end)
        l_ = len(d)
        d = d + [pad] * (l - l_)
        return d, l_


def batchIter(batch_size, data, sN, sL, qL, stop_id=2):
    N = len(data)
    steps = np.ceil(N / float(batch_size))
    steps = int(steps)
    yield steps

    P = np.zeros([batch_size, sN, sL], dtype=np.int32)
    Q = np.zeros([batch_size, qL], dtype=np.int32)
    A = np.zeros([batch_size, sN], dtype=np.int32)

    p_len = np.zeros([batch_size, sN], dtype=np.int32)
    q_len = np.zeros([batch_size], dtype=np.int32)

    for idx in range(steps):
        start = idx * batch_size
        end = (idx + 1) * batch_size
        if end > N:
            batch = data[start:] + data[:end - N]
        else:
            batch = data[start:end]

        P.fill(0)
        Q.fill(0)
        A.fill(0)
        p_len.fill(0)
        q_len.fill(0)

        for i, sample in enumerate(batch):
            sens, q, aid = sample
            for j, s in enumerate(sens[:sN]):
                P[i,j], p_len[i,j] = _transform(s, sL, stop_id)
            Q[i], q_len[i] = _transform(q, qL, stop_id)
            if aid < sN:
                A[i][aid] = 1

        yield idx, P, p_len, Q, q_len, A


def _load(_fname):
    D = []
    with open(_fname, 'r') as f:
        line = f.readline()
        while line:
            n = int(line)
            sen = []
            for i in range(n):
                new = f.readline().strip('\n').split()
                new = map(int, new)
                sen.append(new)
            que = f.readline().strip('\n').split()
            que = map(int, que)
            aid = int(f.readline())
            D.append( [sen, que, aid] )
            f.readline()
            line = f.readline() 
    return D

def load_data(data_dir='/home/zijia/nlp/proj/mult-attention/data/squad/', batch_size=64, vocab_size=50000,
              sN=10, sL=50, qL=15,
              shuffle=True, split_rate=0.9,
              stop_id=2, data_size=None):
    # (TODO) use dev data
    train_ids_path = os.path.join(data_dir, 'ids_vocab%d_train.txt' % vocab_size)
    data = _load(train_ids_path)

    if data_size:
        data = data[:data_size]

    if shuffle:
        random.shuffle(data)
    part = int(np.floor(len(data) * split_rate))
    train = data[:part]
    validate = data[part:]

    titer = batchIter(batch_size, train, sN, sL, qL, stop_id=stop_id)
    viter = batchIter(batch_size, validate, sN, sL, qL, stop_id=stop_id)

    tstep = titer.next()
    vstep = viter.next()

    return titer, tstep, viter, vstep


if __name__ == '__main__':
    process_data(cap=50000)
