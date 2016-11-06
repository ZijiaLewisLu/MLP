import json
import re
import six
import numpy as np
TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)

def format_data(js):
    """
    stat -> { title: (# paragraph, # qas)}
    pairs -> [ [context, [(q,a),..]], .. ]
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
                qas.append( (q.lower().strip(), a[0]['text'].lower().strip()) ) # pairs of unicode strings
            qa_count += len(qas)
            pairs.append([context, qas])
        
        stat[article['title']] = ( len(article['paragraphs']), qa_count)
    return pairs, stat

def create_vocab(pairs, cap=None):
    token_filter = TOKENIZER_RE
    X = [ c + ' '.join([ q+' '+a for q,a in qas ])  for c, qas in pairs ]
    X = '  '.join(X)
    X = token_filter.findall(X)

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
    for p, qas in data:
        p = TOKENIZER_RE.findall(p)
        if len(p) > pl: pl = len(p)
        for q,a in qas:
            q = TOKENIZER_RE.findall(q)
            a = TOKENIZER_RE.findall(a)
            if len(q)>ql: ql=len(q)
            if len(a)>al: al=len(a)
            triple.append([p,q,a])
                
    if qLen is None or ql<qLen: qLen = ql
    if pLen is None or pl<pLen: pLen = pl
    if aLen is None or al<aLen: aLen = al
    with open(voca_path, 'r') as f:
        voca = json.load(f)
    q_np = []
    p_np = []
    a_np = []
    for p,q,a in triple:
        p_np.append(_transform_string(p,voca,pLen))
        q_np.append(_transform_string(q,voca,qLen))
        a_np.append(_transform_string(a,voca,aLen))
    p_np = np.stack(p_np)
    q_np = np.stack(q_np)
    a_np = np.stack(a_np)
    return p_np, q_np, a_np
