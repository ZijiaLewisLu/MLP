import json
# import re
import six
import numpy as np
# TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
from nltk.tokenize import RegexpTokenizer 
word_tokenize = RegexpTokenizer(r'\w+').tokenize

def format_data(js):
    """
    stat -> { title: (# paragraph, # qas)}
    pairs -> [ [context, [(q,a,start),..]], .. ]
    """
    data_list = js['data']
    pairs = []
    stat = {}
    for article in data_list:
        qa_count = 0
        for passage in article['paragraphs']:
            context = passage['context'].lower().strip() # unicode string
            qas = []
            for qa in passage['qas']:
                q = qa['question'] # unicode string
                a = qa['answers']  # list of dicts
                assert len(a)==1, a
                qas.append( (q.lower().strip(), a[0]['text'].lower().strip(), int(a[0]['answer_start'])) ) # pairs of unicode strings
            qa_count += len(qas)
            pairs.append([context, qas])
        
        stat[article['title']] = ( len(article['paragraphs']), qa_count)
    return pairs, stat

def create_vocab(pairs, cap=None):
    X = [ c + ' '.join([ q+' '+a for q,a,s in qas ])  for c, qas in pairs ]
    X = '  '.join(X)
    X = word_tokenize(X)

    # calculate frequency
    f = {}
    for t in X:
        if t in f: f[t] += 1
        else: f[t] = 1

    if cap is not None and cap < len(f):
        print cap
        f = sorted( sorted(six.iteritems(f), key=lambda x: (isinstance(x[0], str), x[0])), 
                    key=lambda x: x[1], reverse=True)[:cap]
    else:
        f = f.items()
            
    #build vocab 
    voca = {"<PAD>":0 , "<UNK>":1, "<STOP>":2}
    add = len(voca)
    for i, p in enumerate(f):
        voca[p[0]] = i+add
    return voca

def _transform_string(s,v,length):
    unk = v['<UNK>']
    Bian = np.zeros(length)
    for i, t in enumerate(s):
        Bian[i] = v[t] if t in v else unk
    return Bian

def transform(voca_path, data, qLen=None, pLen=None, aLen=None, shuffle=True):
    triple=[]
    ql = 0
    pl = 0
    al = 0
    f = open('transform_error.log','w')
    for index , _ in enumerate(data):
        p, qas = _
        p_ = p
        p = word_tokenize(p)
        if len(p) > pl: pl = len(p)
        for q,a,s in qas:
            q = word_tokenize(q)

            a = word_tokenize(a)
            start = len(word_tokenize(p_[:s]))
            indexs = [ start+i for i in range(len(a)) ]
            try:
                [ p[_] for _ in indexs ]
            except Exception:
                continue
                # print index, e
                # f.write(str(index)+'  '+str(e)+'\n')
            # if test != a: print index, a, '--------------', test; f.write(str(index)+'\n')

            if len(q)>ql: ql=len(q)
            if len(a)>al: al=len(a)
            triple.append([p,q,indexs])         
    if qLen is None or ql<qLen: qLen = ql
    if pLen is None or pl<pLen: pLen = pl
    if aLen is None or al<aLen: aLen = al
    with open(voca_path, 'r') as f:
        voca = json.load(f)
    q_np = []
    p_np = []
    a_idx = []
    for ni in np.random.shuffle(range(len(triple))):
        p,q,aid = triple[ni]
        p_np.append(_transform_string(p,voca,pLen))
        q_np.append(_transform_string(q,voca,qLen))

        aid.append(pLen)
        a_idx += [  [ni,t,wid] for t, wid in enumerate(aid) ]

    p_np = np.stack(p_np).astype(np.int)
    q_np = np.stack(q_np).astype(np.int)
    
    sparse_answer = [ [1]*len(a_idx), a_idx, [len(a_idx), aLen+1, pLen+1] ] # values, idx, shape
    return p_np, q_np, sparse_answer

def load_data(js='./squad/train-v1.1.json', voca='./squad/voca_82788.json'):
    with open(js,'r') as f:
        squad = json.load(f)
    pair, stat = format_data(squad)
    p,q,a = transform(voca, pair)
    return p, q, a

if __name__ == '__main__':
    voca = './squad/voca_82788.json'
    with open('./squad/train-v1.1.json') as f:
        squad = json.load(f)
    pair, stat = format_data(squad)
    transform(voca, pair)