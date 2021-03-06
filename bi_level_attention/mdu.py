import json
import pickle as pk
import numpy as np
import random
import os
from tqdm import tqdm
import re
from glob import glob
from nltk import TreebankWordTokenizer
from string import punctuation

from collections import Counter

_tokenrize = TreebankWordTokenizer().tokenize


_START_VOCAB = ["<unk>", "<pad>", "<stop>"]


def format_data(js):
    data_list = js['data']
    formated = []
    for article in data_list:
        for passage in article['paragraphs']:
            context = passage['context'].strip()  # unicode string
            for qa in passage['qas']:
                q = qa['question']  # unicode string
                # a = qa['answers']   # list of dicts
                answer = [(_['text'].strip(), int(_['answer_start']))
                          for _ in qa['answers']]
                answer = set(answer)
                formated.append((
                    context,
                    q.strip(),
                    answer))
    return formated


def filter_data(formated, exps=u"([^A-Z]{2,6})([.?!;]+)(\s+[A-Z]\w*|$)", pos_idx=2):
    def _locate(c, a, s, period_loc):
        if c[s] != a[0]:
            return None, 1
        for i in range(len(period_loc) - 1):
            h = period_loc[i]
            t = period_loc[i + 1]
            if h <= s and s < t:
                if s + len(a) > t:
                    return i, 2
                else:
                    return i, 0
        return None, 3

    def _one_piece(_data):
        c, q, a_s = _data
        # get delimiter location
        period_loc = [_.start(pos_idx) + 1 for _ in re.finditer(exps, c)]
        period_loc.insert(0, 0)
        if period_loc[-1] != len(c):
            period_loc.append(len(c))

        # get answer location
        _cate = [[] for _ in range(4)]
        for a, s in a_s:
            loc, i = _locate(c, a, s, period_loc)
            _cate[i].append(loc)
        for i in range(4):
            if len(_cate[i]) > 0:
                _cate[i] = [c, q, a_s, period_loc, _cate[i]]
        return _cate

    # =============================
    catego = [[] for _ in range(4)]  # good, bad, abrevation, mysterious
    for _ in tqdm(formated):
        rslt = _one_piece(_)
        assert len(rslt) == 4, rslt
        for i in range(4):
            if len(rslt[i]) > 0:
                catego[i].append(rslt[i])

    print "#good:%d #bad_sample:%d #confusing abbrevation:%d #mysterious:%d #all:%d" \
        % (len(catego[0]), len(catego[1]), len(catego[2]), len(catego[3]), len(formated))
    return catego

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"(?<=\d),(?=\d)", '', string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\'", " \' ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def token(words):
    words = clean_str(words)
    ts = _tokenrize(words)
    # ts = [ x.strip(punctuation) for x in ts ]
    ts = [ x for x in ts if len(x)>0 ]
    return ts

def token_sample(data, normalize_digit=True):
    c, q, a_s, period_loc, asi = data
    sentence = []
    for i in range(len(period_loc)-1):
        s = c[period_loc[i]:period_loc[i+1]].strip('. ').lower()
        if normalize_digit:
            s = re.sub('\d', '0', s)
        tk = token(s)
        if len(tk) > 0:
            sentence.append(tk)

    q = q.strip(' ?').lower()
    if normalize_digit:
        q = re.sub('\d', '0', q)
    q = token(q)

    answers = []
    for a,s in a_s:
        a = a.strip().lower()
        if normalize_digit:
            a = re.sub('\d', '0', a)
        a = token(a)
        answers.append(a)

    return [sentence, q, asi, answers]

# @DeprecationWarning
# def token_data(origin_file, save_name, normalize_digit=True):
#     if os.path.exists(save_name):
#         with open(save_name, 'r') as f:
#             formated = json.load(f)
#     else:
#         with open(origin_file, 'r') as f:
#             squad = json.load(f)
#         fours = format_data(squad)
#         good, bad, abr, mys = filter_data(fours)
#         formated = [token_sample(_, normalize_digit=normalize_digit)
#                     for _ in good]
#         with open(save_name, 'w') as f:
#             json.dump(formated, f, indent=4)
#     return formated


def create_vocab(tokens, cap=None):
    X = []
    for sentence, q, asi, answers in tokens:
        X.extend(q)
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
    return vocab, f

def restruct_glove_words(fname):
    words = []
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip('\n').split(' ')[0]
            words.append(line)
    return words


def restruct_glove_embedding(fname, vocab, dim=300):
    V = len(vocab)
    done = set()
    embedding = np.zeros([V, dim])
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            word = line[0]
            if word in vocab:
                ID = vocab[word]
                embedding[ID] = map(float, line[1:])
                done.add(ID)
    assert len(done) == V
    return embedding


def create_vocab_glove(data_triple, glove_words, cap=None):
    X = []
    for sentence, q, asi, answers in data_triple:
        X.extend(q)
        for s in sentence:
            X.extend(s)

    f = {}
    for t in X:
        if t in f:
            f[t] += 1
        else:
            f[t] = 1

    vocab_list = ['<unk>'] + sorted(f, key=f.get, reverse=True)
    glove_words = set(glove_words)
    i = 0
    condition = lambda x: (x < len(vocab_list) and x <
                           cap) if cap else (x < len(vocab_list))
    while condition(i):
        k = vocab_list[i]
        if k not in glove_words:
            vocab_list.pop(i)
        else:
            i += 1
    if cap:
        vocab_list = vocab_list[:cap]

    vocab = {k: i for i, k in enumerate(vocab_list)}
    return vocab

# Tf Idf =========================================================

def idf(documents):
    v={}
    for d in documents:
        d = set(d)
        for tk in d:
            if tk in v:
                v[tk] += 1
            else:
                v[tk] = 1
        
    D = len(documents)
    _idf = { k:np.log(D/float(ct)) for k, ct in v.items() }
    return _idf

def token2cst(tk, constant=1):
    ones = []
    for sens, q, sid, a in tk:
        scst = [ [constant]*len(_) for _ in sens ]
        qcst = [constant]*len(q)
        ones.append( [scst, qcst] )
    return ones

def token2idf(tk, _idf):
    idf = []
    cvt = lambda x: _idf.get(x,0.0)
    for sens, q, sid, a in tk:
        sens_idf = []
        for s in sens:
            sens_idf.append( map( cvt , s) )
        q_idf = map( cvt, q )
        idf.append( [sens_idf, q_idf] )
    return idf

def token2tfidf(tk, _idf):
    tfidf = []
    for sens, q, sid, a in tk:
        all_token = list(q)*3
        for s in sens:
            all_token += s
        ct = Counter(all_token)
        
        cvt = lambda x : ct[x] * _idf.get(x, 0)
        
        sens_idf = []
        for s in sens:
            sens_idf.append( map( cvt , s) )
        q_idf = map( cvt, q )
        tfidf.append( [sens_idf, q_idf] )
    return tfidf

# word ids ==============================================

def _t2id(tlist, v, unk_id=0):
    ids = [v.get(t, unk_id) for t in tlist]
    return ids

def tokens2id(data, vocab, unk="<unk>"):
    unk_id = vocab[unk]
    ids = []
    for sen, qu, sid, answers in data:
        sen_id = [_t2id(_, vocab, unk_id) for _ in sen]
        qu_id = _t2id(qu, vocab, unk_id)
        answer_id = [_t2id(_, vocab, unk_id) for _ in answers]
        ids.append([sen_id, qu_id, sid, answer_id])
    return ids

def reverse(IDS, re_vocab):
    return " ".join( [ re_vocab.get(_) for _ in IDS] )

# def process_data(data_path='../data/squad', save_path='../data/squad/',
# cap=None, normalize_digit=True):

#     train_js = glob(os.path.join(data_path, 'train-v*.json'))[0]
#     dev_js = glob(os.path.join(data_path, 'dev-v*.json'))[0]

#     # token data
#     formated_save_path = os.path.join(save_path, 'formated')
#     train_token = token_data(
#         train_js, formated_save_path + '_train.js', normalize_digit=normalize_digit)
#     dev_token = token_data(dev_js, formated_save_path +
#                            '_dev.js', normalize_digit=normalize_digit)

#     # load vocab
#     if cap is None:
#         vocab_path = os.path.join(save_path, 'vocab_full.js')
#     else:
#         vocab_path = os.path.join(save_path, 'vocab_%d.js' % cap)
#     if os.path.exists(vocab_path):
#         with open(vocab_path, 'r') as f:
#             vocab = json.load(f)
#         print 'Vocabulary loaded'
#     else:
#         vocab = create_vocab(formated, cap=cap)
#         with open(vocab_path, 'w') as f:
#             json.dump(vocab, f, indent=4)
#         print 'Vocabulary created'

#     # transform data
#     ids_path = os.path.join(save_path, 'ids_vocab%d' % len(vocab))
#     train_ids = data2id(train_token, vocab)
#     dev_ids = data2id(dev_token, vocab)
#     _save(ids_path + '_train.txt', train_ids)
#     _save(ids_path + '_dev.txt', dev_ids)


def _transform(d, l, end, pad=1, add_end=True):
    d = list(d)
    if len(d) >= l:
        if add_end:
            d[l - 1] = end
        return d[:l], l
    else:
        if add_end:
            d.append(end)
        l_ = len(d)
        d = d + [pad] * (l - l_)
        return d, l_


def batchIter(batch_size, data, idf, sN=10, sL=50, qL=15, stop_id=2, add_stop=True):
    N = len(data)
    steps = np.ceil(N / float(batch_size))
    steps = int(steps)
    yield steps

    P = np.ones([batch_size, sN, sL], dtype=np.int32)
    Q = np.ones([batch_size, qL], dtype=np.int32)
    A = np.zeros([batch_size, sN], dtype=np.int32)

    P_idf = np.zeros([batch_size, sN, sL], dtype=np.float32)
    Q_idf = np.zeros([batch_size, qL], dtype=np.float32)

    p_len = np.zeros([batch_size, sN], dtype=np.int32)
    q_len = np.zeros([batch_size], dtype=np.int32)

    for idx in range(steps):
        start = idx * batch_size
        end = (idx + 1) * batch_size
        if end > N:
            batch_data = data[start:] + data[:end - N]
            batch_idf  = idf[start:] + idf[:end - N]
        else:
            batch_data = data[start:end]
            batch_idf = idf[start:end]

        P.fill(1)
        Q.fill(1)
        A.fill(0)
        P_idf.fill(0)
        Q_idf.fill(0)
        p_len.fill(0)
        q_len.fill(0)

        for i in range(batch_size):
            sens, q, sid, answer = batch_data[i]
            senidf, qidf = batch_idf[i]
            for j in range(min(sN, len(sens))):
                P[i, j], p_len[i, j] = _transform(sens[j], sL, stop_id, pad=1, add_end=add_stop)
                P_idf[i,j], _ = _transform(senidf[j], sL, None, pad=0, add_end=False)

            Q[i], q_len[i] = _transform(q, qL, stop_id, pad=1, add_end=add_stop)
            Q_idf[i], _ = _transform(qidf, qL, None, pad=0, add_end=False)
            
            for a in sid:
                if a < sN:
                    A[i][a] = 1

        yield idx, P, P_idf, p_len, Q, Q_idf, q_len, A

def id_save(_fname, _data):
    with open(_fname, 'w') as f:
        for seq, qur, sid, answer_id in _data:
            N = len(seq)
            f.write("%d\n" % N)
            for s in seq:
                f.write(" ".join(map(str, s)) + '\n')
            f.write(" ".join(map(str, qur)) + '\n')
            f.write(" ".join(map(str, sid)) + '\n')
            A = len(answer_id)
            f.write('%d\n' % A)
            for a in answer_id:
                f.write(" ".join(map(str, a))+'\n')
            f.write('\n')


def id_load(_fname):
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
            
            sid = f.readline().strip('\n').split()
            sid = map(int, sid)

            a = int(f.readline().strip('\n'))
            answer = []
            for i in range(a):
                new = f.readline().strip('\n').split()
                new = map(int, new)
                answer.append(new)

            D.append([sen, que, sid, answer])
            f.readline()
            line = f.readline()
    return D

def weight_save(fname, data):
    f = open(fname, 'w')
    for sp in data:
        sen, que = sp
        f.write('%d\n'%len(sen))
        _sen = [ ' '.join( map(str, _) ) for _ in sen ]
        f.write('\n'.join(_sen))
        f.write('\n\n')
        f.write(' '.join(map(str,que)))
        f.write('\n\n')
    f.close()

def weight_load(fname):
    f = open(fname, 'r')
    data = []
    line = f.readline()
    while line:
        n = int(line.strip('\n'))
        sen = []
        for i in range(n):
            new = f.readline().strip('\n').split(' ')
            new = map(float, new)
            sen.append(new)
        f.readline()
        que = f.readline().strip('\n').split(' ')
        que = map(float, que)
        data.append([sen, que])
        f.readline()
        line = f.readline()
    f.close()
    return data

def prepare_data(id_path, wt_path, data_size=None, size=3185, val_rate=0.05):
    train_data = id_load(id_path)
    train_wt   = weight_load(wt_path)

    validate_data = train_data[-size:]
    validate_wt  = train_wt[-size:]
    train_data = train_data[:-size]
    train_wt = train_wt[:-size]

    if data_size:
        train_data = train_data[:data_size]
        train_wt  = train_wt[:data_size]
    vsize = max(20, len(train_data) * val_rate)
    vsize = int(min(vsize, len(validate_data)))
    return train_data, train_wt, validate_data, validate_wt, vsize


# def load_data(data_dir='/home/zijia/nlp/proj/mult-attention/data/squad/', batch_size=64, vocab_size=50000,
#               sN=10, sL=50, qL=15,
#               shuffle=True, split_rate=0.9,
#               stop_id=2, data_size=None):
#     # (TODO) use dev data
#     train_ids_path = os.path.join(
#         data_dir, 'ids_vocab%d_train.txt' % vocab_size)
#     data = _load(train_ids_path)

#     if data_size:
#         data = data[:data_size]

#     if shuffle:
#         random.shuffle(data)
#     part = int(np.floor(len(data) * split_rate))
#     train = data[:part]
#     validate = data[part:]

#     titer = batchIter(batch_size, train, sN, sL, qL, stop_id=stop_id)
#     viter = batchIter(batch_size, validate, sN, sL, qL, stop_id=stop_id)

#     tstep = titer.next()
#     vstep = viter.next()

#     return titer, tstep, viter, vstep


if __name__ == '__main__':
    # process_data(cap=50000)
    with open('./data/squad/train-v1.1.json') as f:
        squad = json.load(f)
    with open('./data/squad/vocab_not_glove_60000.js') as f:
        vocab = json.load(f)

    formated = format_data(squad)
    gp = filter_data(formated)
    good = gp[0]
    print len(good)

    tokens = [ token_sample(_) for _ in good ]  
    ids = tokens2id(tokens, vocab)

    wt_dc = []
    for sen, que, sid, ans in tokens:
        d = list(que)*3
        for s in sen:
            d += s

    idf_map = idf(wt_dc)
    ones = token2cst(tokens)
    idf  = token2idf(tokens, idf_map)
    tfidf = token2tfidf(tokens, idf_map)

    id_save('./data/squad/ids_not_glove60000_train.txt', ids)
    weight_save('./data/squad/train_ones.txt', ones)
    weight_save('./data/squad/train_idf.txt', idf)
    weight_save('./data/squad/train_tfidf.txt', tfidf)