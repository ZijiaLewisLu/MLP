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
    p_np = np.stack(p_np).astype(np.int)
    q_np = np.stack(q_np).astype(np.int)
    a_np = np.stack(a_np).astype(np.int)

    order = np.random.shuffle(range(p_np.shape[0]))
    p_np = p_np[order][0]
    q_np = q_np[order][0]
    a_np = a_np[order][0]
    
    sparse_answer = locate_answer(p_np,a_np)
    return p_np, q_np, sparse_answer

def conform(a,b,i,j):
    if i == len(a):
        return True
    if j == len(b):
        return False
    if a[i] != b[j]:
        return False
    if a[i] == b[j]:
        return conform(a,b,i+1,j+1)
    
def locate_answer(plist,alist):
    shape = [alist.shape[1]+1, plist.shape[1]+1] # T+1,Q+1
    answers = []
    for n in range(len(plist)):
        idx = []
        sp = plist[n]; sa = alist[n]
        i_a =0; i_p = 0
        while i_a < len(sa) and sa[i_a]!=0: i_a +=1
        while i_p < len(sp) and sp[i_p]!=0: i_p +=1
        sa = sa[:i_a]; sp = sp[:i_p]
        start = []
        for i in range(len(sp)):
            if conform(sa, sp, 0, i):
                start.append(i)
        for t in range(i_a):
            idx += [ [t,s+t] for s in start ]
        idx.append( [i_a,shape[1]-1] )
        value = [1]*len(idx)
        answers.append([idx,value,shape])
    return answers

def load_data():
    with open('./squad/train-v1.1.json') as f:
        squad = json.load(f)
    pair, stat = format_data(squad)
    p,q,a = transform('./squad/vocab_99036.json', pair)
    return p, q, a

if __name__ == '__main__':
    load_data()
