import json
import six
import numpy as np
from nltk.tokenize import RegexpTokenizer
word_tokenize = RegexpTokenizer(r'\w+').tokenize


def format_data(js):
    """
    stat -> { title: (# paragraph, # qas)}
    pairs -> [ (context,q,a,start),.. ]
    """
    data_list = js['data']
    fours = []
    # stat = {}
    for article in data_list:
        # qa_count = 0
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
        # stat[article['title']] = (len(article['paragraphs']), qa_count)
    return fours


def create_vocab(pairs, cap=None):
    X = [c + ' '.join([q + ' ' + a for q, a, s in qas]) for c, qas in pairs]
    X = '  '.join(X)
    X = word_tokenize(X)

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
                   key=lambda x: x[1], reverse=True)[:cap]
    else:
        f = f.items()

    # build vocab
    voca = {"<PAD>": 0, "<UNK>": 1, "<STOP>": 2}
    add = len(voca)
    for i, p in enumerate(f):
        voca[p[0]] = i + add
    return voca


def _transform_string(s, v, length):
    unk = v['<UNK>']
    Bian = np.zeros(length)
    for i, t in enumerate(s):
        Bian[i] = v[t] if t in v else unk
    return Bian


def transform(voca_path, data, qLen=None, pLen=None, aLen=None):
    triple = []
    ql = 0
    pl = 0
    al = 0
    # f = open('transform_error.log', 'w')
    for index, _ in enumerate(data):
        p, q, a, s = _

        start = len(word_tokenize(p[:s]))
        a = word_tokenize(a)
        indexs = [start + i for i in range(len(a))] #

        p = word_tokenize(p)
        q = word_tokenize(q)
        try:
            [p[_] for _ in indexs]
        except Exception:
            continue
                # print index, e
                # f.write(str(index)+'  '+str(e)+'\n')
            # if test != a: print index, a, '--------------', test;
            # f.write(str(index)+'\n')
        if len(p) > pl:
            pl = len(p)
        if len(q) > ql:
            ql = len(q)
        if len(a) > al:
            al = len(a)
        triple.append([p, q, indexs])

    if qLen is None or ql < qLen:
        qLen = ql
    if pLen is None or pl < pLen:
        pLen = pl
    if aLen is None or al < aLen:
        aLen = al
    with open(voca_path, 'r') as f:
        voca = json.load(f)
    q_np = []
    p_np = []
    a_np = []

    # order = range(len(triple))
    for p, q, aid in triple:
        p_np.append(_transform_string(p, voca, pLen))
        q_np.append(_transform_string(q, voca, qLen))

        aid.append(pLen)
        a_new = np.zeros([aLen + 1])
        a_new[:len(aid)] = aid
        a_np.append(a_new)

    p_np = np.stack(p_np).astype(np.int)  # N,pLen
    q_np = np.stack(q_np).astype(np.int)
    a_np = np.stack(a_np).astype(np.int)  # N,aLen+1
    return p_np, q_np, a_np


def load_data(js='./squad/train-v1.1.json', voca='./squad/voca_82788.json'):
    with open(js, 'r') as f:
        squad = json.load(f)
    fours = format_data(squad)
    p, q, a = transform(voca, fours)
    return p, q, a


def batchIter(batch_size, datas, shuffle=True):
    if not isinstance(datas, list):
        datas = [datas]      
    
    N = len(datas[0])
    order = range(N)
    if shuffle:
        np.random.shuffle(order)
        order = list(order)

    step = N / batch_size + 1
    for i in range(step):
        start = i * batch_size
        end = (i + 1) * batch_size
        if end > N:
            batch_idx = order[start:]+order[:end-N]
        else:
            batch_idx = order[start:end]

        batch = [ d[batch_idx] for d in datas ]
        if len(batch) == 1:
            batch = batch[0] 
        yield batch


if __name__ == '__main__':
    # voca = './squad/voca_82788.json'
    # with open('./squad/train-v1.1.json') as f:
    #     squad = json.load(f)
    # pair, stat = format_data(squad)
    # transform(voca, pair)

    d = np.array(range(20))
    iters = batchIter(7,d, shuffle=False)
    for _ in iters:
        print _
