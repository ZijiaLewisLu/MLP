import numpy as np
# import tensorflow as tf
from main import prepare_data
from mdu import batchIter
import os
from glob import glob
import matplotlib.pyplot as plt

batch_size = 32
sN=10
sL=50
qL=15
vocab_size=50000
size=256

def one_step(_iter, model, sess, fetch=None):
    if fetch is None:
        fetch = model.score
    batch_idx, P, p_len, Q, q_len, A = _iter.next()
    rslt = sess.run(fetch, feed_dict={
                            model.passage: P,
                            # model.p_len: p_len,
                            model.query: Q,
                            # model.q_len: q_len,
                            model.answer: A,
                            model.dropout: 1.0,
                                })
    return [batch_idx, P, p_len, Q, q_len, A], rslt

def softmax(x, axis=-1):
    # x -= x.mean()
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)

def cross_entropy_loss(logit, label, axis=-1):
    logit = softmax(logit)
    log = np.log(logit)
    return label * log

def evaluate(model, sess, step=50, check=False, sN=sN, sL=sL, qL=qL, batch_size=batch_size):
    train, val, vsize = prepare_data('./data/squad/ids_vocab50000_train.txt')
    np.random.shuffle(train)
    np.random.shuffle(val)
    titer = batchIter(batch_size, train, sN, sL, qL, stop_id=2)
    viter = batchIter(batch_size, val,   sN, sL, qL, stop_id=2)
    titer.next()
    viter.next()
    
    def _eval(_iter):
        pdis = np.zeros(10)
        adis = np.zeros(10)
        acc = 0.0
        loss = 0.0
        for i in range(step):
            try:
                data, rslt = one_step(_iter, model, sess, [model.score, model.loss, model.accuracy] )
                loss+= rslt[1].mean()
                acc += rslt[2]
            except StopIteration:
                print 'stop at round %d, out of data' % i
                break  

            prediction = np.argmax(rslt[0], axis=1)
            for j in prediction:
                pdis[j] += 1
            label = np.argmax(data[-1],axis=1)
            for j in label:
                adis[j] += 1

            if check:
                logit  = rslt[0]
                answer = data[-1]
                _loss = cross_entropy_loss(logit, answer)
                if not _loss.mean() == rslt[1].mean():
                    # print 'Loss Error', _loss, rslt[1]
                    # print "loss error"
                    pass
                _acc = np.mean( prediction == label )
                if not _acc == rslt[2]:
                    print 'Accuracy Error', _acc, rslt[2]

        pdis /= pdis.sum()
        adis /= adis.sum()
        print "accuracy: %.4f, loss: %.4f" % (acc/(i+1), loss/(i+1))
        random_acc = np.sum( pdis*adis )
        print "random prediction acc: %.4f" % random_acc  
        return pdis, adis, acc, loss

    print '======= train set ========'
    t = _eval(titer)
    print '======= dev set =========='
    v = _eval(viter)
    return t+v


def concat_attention(p_rep, q_rep, Wp, Wq, Ws):
    x = np.dot(p_rep, Wp) + np.dot(q_rep, Wq)
    x = np.tanh(x)
    x = x.dot(Ws)
    return x

def mock_attention(sess):
    pass

def parse_track_log(log_dir):
    trackfile = os.path.join(log_dir, 'tracking.log')
    # track_dir = os.path.join(log_dir, 'track')
    with open(trackfile, 'r') as f:
        content = f.readlines()

    for i, c in enumerate(content):
        time, info = c.strip('\n ').split(' INFO ')
        content[i] = [time, info]

    return content

def load_at_mark(mark, track_dir):
    # train
    import numpy as np
    def _one(name):
        data = []
        files = glob( os.path.join(track_dir, name) )
        files = sorted(files)
        for f in files:
            data.append(np.load(f))
        return data

    suffics = ['Tacc*', 'Tloss*', 'Vacc*', 'Vloss*']
    suffics = [ '%s_%s'%(mark, _) for _ in suffics  ]
    D = [ _one(_) for _ in suffics ]
    return D

def visualize_attention(alignment):
    plt.imshow(alignment, interpolation='none')
    plt.grid()
    plt.colorbar(orientation='vertical')
    plt.show()
