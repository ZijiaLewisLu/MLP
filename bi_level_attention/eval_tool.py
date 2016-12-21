# /usr/bin/python

import numpy as np
import tensorflow as tf
from mdu import batchIter, prepare_data
import os
from glob import glob
import matplotlib.pyplot as plt
from collections import namedtuple
import json
import pickle as pk
from mdu import _load
import sys

batch_size = 128
sN = 10
sL = 50
qL = 15
vocab_size = 50000
size = 256

Param = namedtuple('Param', 'data result')


def read(fname):
    if os.path.isdir(fname):
        fname = os.path.join(fname, 'Flags.js')
    with open(fname, 'r') as f:
        flag = json.load(f)
    return flag


def from_log(log_folder):
    print read(log_folder)
    ck_path = os.path.join(log_folder, '/ckpts')
    ck = tf.train.get_checkpoint_state(ck_path)
    ckfile = ck.model_checkpoint_path
    meta = ckfile + '.meta'
    return ck, meta


def one_step(_iter, model, sess, fetch=None):
    if fetch is None:
        fetch = [model.alignment, model.loss, model.accuracy]
    batch_idx, P, p_len, Q, q_len, A = _iter.next()
    rslt = sess.run(fetch, feed_dict={
        model.passage: P,
        # model.p_len: p_len,
        model.query: Q,
        # model.q_len: q_len,
        model.answer: A,
        model.dropout: 1.0,
    })

    return Param(data=[batch_idx, P, p_len, Q, q_len, A],
                 result=rslt)

def concat_attention(p_rep, q_rep, Wp, Wq, Ws):
    x = np.dot(p_rep, Wp) + np.dot(q_rep, Wq)
    x = np.tanh(x)
    x = x.dot(Ws)
    return x


def mock_attention(sess):
    pass


def norm(x):
    if not isinstance(x, np.ndarray):
        x = x.values
    return np.sqrt((x**2).sum())


def softmax(x, axis=-1):
    x -= x.max()
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def cross_entropy_loss(logit, label, axis=-1):
    logit = softmax(logit)
    log = np.log(logit)
    return label * log


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
        files = glob(os.path.join(track_dir, name))
        files = sorted(files)
        for f in files:
            data.append(np.load(f))
        return data

    suffics = ['Tacc*', 'Tloss*', 'Vacc*', 'Vloss*']
    suffics = ['%s_%s' % (mark, _) for _ in suffics]
    D = [_one(_) for _ in suffics]
    return D


def visualize_attention(alignment):
    plt.imshow(alignment, interpolation='none')
    plt.grid()
    plt.colorbar(orientation='vertical')
    plt.show()


def choose_ckpt(ckpt_dir):
    ck = tf.train.get_checkpoint_state(ckpt_dir)
    ckfiles = ck.all_model_checkpoint_paths[::-1]

    for i, p in enumerate(ckfiles):
        print "%d: %s" % (i, p)
    idx = raw_input('Choose which to load ')

    if len(idx) == 0:
        files = [ck.model_checkpoint_path]
    else:
        idx = map(int, idx.strip(' ').split(' '))
        files = [ckfiles[_] for _ in idx]

    return files


def dig_out(g):
    op_names = [_.name for _ in g.get_operations()]

    def get(name):
        if not name.startswith('model/'):
            name = 'model/' + name
        return g.get_tensor_by_name(name + ':0')

    if 'Softmax' in op_names:
        alignment = get('Softmax')
    else:
        alignment = get('alignment')

    for k in op_names:
        if k.endswith('/attention'):
            score = get(k)
            break

    train_op = None
    if 'model/Adam' in op_names:
        train_op = g.get_operation_by_name('model/Adam')

    params = {
        'loss': get('loss'),
        'accuracy': get('accuracy'),
        'alignment': alignment,
        'score': score,
        'passage': get('passage'),
        'p_len': get('p_len'),
        'q_len': get('q_len'),
        'p_idf': get('p_idf'),
        'q_idf': get('q_idf'),
        'dropout': get('dropout_rate'),
        'query': get('query'),
        'answer': get('answer'),
        'train_op': train_op,
    }
    return params

def evaluate(sess, tensor_dict,
             step=100, check=False, feed_func=None,
             train_op=None,
             sN=sN, sL=sL, qL=qL, batch_size=batch_size,
             train_ids_path='./data/squad/ids_not_glove60000_train.txt',
             train_idf_path='./data/squad/train_tfidf.pk',
             dev_ids_path='./data/squad/ids_not_glove60000_dev.txt',
             dev_idf_path='./data/squad/dev_tfidf.pk',
             ):

    train_data, train_idf, validate_data, validate_idf, vsize \
        = prepare_data(train_ids_path, train_idf_path)

    dev_data = _load(dev_ids_path)
    with open(dev_idf_path, 'r') as f:
        dev_idf = pk.load(f)

    # np.random.shuffle(train)
    # np.random.shuffle(val)
    # titer = batchIter(batch_size, train_data, train_idf, sN, sL, qL, stop_id=2)
    viter = batchIter(batch_size, validate_data,
                      validate_idf, sN, sL, qL, stop_id=2)
    diter = batchIter(batch_size, dev_data, dev_idf, sN, sL, qL, stop_id=2)

    fetch = ['loss', 'accuracy', 'score', 'alignment']
    fetch = [tensor_dict[_] for _ in fetch]

    if feed_func is None:
        def feed_func(_data):
            batch_idx, P, p_idf, p_len, Q, q_idf, q_len, A = _data
            return {tensor_dict['passage']: P,
                    tensor_dict['query']: Q,
                    tensor_dict['p_len']: p_len,
                    tensor_dict['q_len']: q_len,
                    tensor_dict['p_idf']: p_idf,
                    tensor_dict['q_idf']: q_idf,
                    tensor_dict['answer']: A,
                    tensor_dict['dropout']: 1.0}

    def _eval(_iter, dev=False):

        if dev:
            fetch = tensor_dict['alignment']
        else:
            fetch = ['loss', 'accuracy', 'score', 'alignment']
            fetch = [tensor_dict[_] for _ in fetch]

        pdis = np.zeros(10)
        adis = np.zeros(10)
        acc = 0.0
        loss = 0.0
        i = _iter.next()
        for data in _iter:
            rslt = sess.run(fetch, feed_func(data))
            # rslt = rslt[:-1]

            if not dev:
                loss += rslt[0].mean()
                acc += rslt[1]
                prediction = np.argmax(rslt[3], axis=1)
                for j in prediction:
                    pdis[j] += 1
                label = np.argmax(data[-1], axis=1)
                for j in label:
                    adis[j] += 1
            else:
                prediction = rslt.argmax(1)
                answer = data[-1]
                correct = (answer[range(batch_size), prediction] == 1)
                acc += correct.mean()

        print "accuracy: %.4f, loss: %.4f" % (acc / (i + 1), loss / (i + 1))

        if not dev:
            pdis /= pdis.sum()
            adis /= adis.sum()
            random_acc = np.sum(pdis * adis)
            print "random prediction acc: %.4f" % random_acc
            return pdis, adis, acc, loss
        else:
            return None, None, acc, loss

    print '======= dev set =========='
    v = _eval(viter)
    print

    # print '======= train set ========'
    # if train_op is not None:
    #     fetch.append(train_op)
    #     print '  Training'
    # t = _eval(titer)

    print '========test set =========='
    d = _eval(diter, dev=True)
    print '\n\n'

    # return t + v
    return d + v


def main():
    load_path = sys.argv[1]
    gpu = 1
    if len(sys.argv) > 2:
        gpu = int(sys.argv[2])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    with open(os.path.join(load_path, 'Flags.js'), 'r') as f:
        old_flag = json.load(f)

    batch_size = old_flag['batch_size']
    print 'batch_size', batch_size

    with tf.Session() as sess:

        if os.path.isdir(load_path):
            ckfiles = choose_ckpt(os.path.join(load_path, 'ckpts'))
            # assert fname is not None
        else:
            ckfiles = [load_path]

        meta = ckfiles[0] + '.meta'
        saver = tf.train.import_meta_graph(meta)
        print 'Graph imported'

        for idx, fname in enumerate(ckfiles):
            print '%d -- %s' % (idx, fname)
            saver.restore(sess, fname)

            M = dig_out(sess.graph)

            evaluate(sess, M, batch_size=batch_size)


if __name__ == '__main__':
    main()
