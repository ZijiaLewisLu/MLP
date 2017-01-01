#! /usr/bin/python
import os
import tensorflow as tf
from glob import glob
import pickle
from utils import data_to_token_ids, define_gpu
import json
import numpy as np
# from attentive_model import AttentiveReader
from collections import namedtuple, Counter
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import sys

TensorName = ['loss', 'accuracy', 'attention', 'document',
              'query', 'docu-end', 'quer-end', 'Y', 'dropout_rate',
              'score']
# All_Tensor = Tensor_feed + Tensor_fetch
name_attr_map = {
    'loss': 'loss',
    'accuracy': 'accuracy',
    'attention': 'attention',
    'query': 'query',
    'document': 'document',
    'docu-end': 'd_end',
    'quer-end': 'q_end',
    'Y': 'y',
    'dropout_rate': 'dropout',
    'score': 'score'
}

AttrName = map(name_attr_map.get, TensorName)

Tensor = namedtuple('Tensor', ' '.join(AttrName))


def eval_iter(flist, max_nstep, max_query_step, batch_size, vocab):

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

        y.fill(0)
        ds.fill(0)
        qs.fill(0)

        for idx, fname in enumerate(files):
            rslt = data_to_token_ids(fname, None, vocab, save=False)
            document = rslt[2].strip('\n')
            question = rslt[4].strip('\n')
            answer = rslt[6].strip('\n')

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


def dig_tensors(graph, targ=TensorName, _map=name_attr_map):
    tensors = { k: None for k in _map.values()}

    for name in targ:
        tensors[_map.get(name, name)] = graph.get_tensor_by_name(name + ':0')

    tr = Tensor(**tensors)
    return tr


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


def create_revert(revocab):
    def revert(ids, rm_pad=True):
        words = [revocab[_] for _ in ids]
        if rm_pad:
            words = [_ for _ in words if _ != '<PAD>']
        return words
    return revert


def visualize_emb(data, emd, vocab=None, transform=True, tsne=None):

    if transform:
        ids = [vocab[_] for _ in data]
    else:
        ids = data

    rep = emd[ids]
    if tsne is None:
        tsne = TSNE()
    e_space = tsne.fit_transform(rep)
    plt.plot(e_space[:, 1], e_space[:, 0], "ro")
    for index, word in enumerate(data):
        plt.annotate(word, xy=(e_space[index, 1], e_space[index, 0]),
                     xytext=(-5, 5),
                     textcoords='offset points', ha='left', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5',
                               fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


def step(data, M, sess, fetch):
    idx, doc, d_end, que, q_end, y = data
    feed = {M.document: doc,
            M.query: que,
            M.d_end: d_end,
            M.q_end: q_end,
            M.y: y,
            M.dropout: 1.0,
            }
    return sess.run(fetch, feed)


def Topk(array, k):
    top = np.argpartition(-array, k)[:k]
    pair = sorted(zip(array[top], top), key=lambda x: x[0], reverse=True)
    return [_[1] for _ in pair]


def analyse(doc, answer, attention):
    batch_size = doc.shape[0]
    att_pred = attention.argmax(1)
    ap_ids = doc[range(batch_size), att_pred].astype(np.int)

    _common_ids = []
    for i, each_sample in enumerate(attention):
        top = Topk(each_sample, 20)
        top_words = doc[i][top]
        ct = Counter(top_words)
        _common_ids.append(ct.most_common(1)[0][0])
    common_ids = np.array(_common_ids)

    aid = answer.argmax(1)
    # pid = score_or_prob.argmax(1)

    cright = common_ids == aid
    pright = ap_ids == aid
    both = np.logical_or(cright, pright)
    # correct = pid == aid
    return pright, cright, both

def stat(atten, doc, topk):
    top_index = Topk(atten, topk) # indexes
    top_atten = atten[top_index]
    top_wordid = doc[top_index]
    
#     ct = Counter(top_wordid)
    entities = set(top_wordid)
    stat = []
    for e in entities:
        count = 0
        _p = 0.0
        for i, wid in enumerate(top_wordid):
            if wid == e:
                count += 1
                _p += top_atten[i]
        stat.append([ e ,count, _p])
    stat = sorted(stat, key=lambda x: x[2], reverse=True)
    return stat

def extract_and_compare(_doc, score, atten, topk=20):
    N = _doc.shape[0]
    max_right = []
    pid = score.argmax(1)
    for idx in range(N):
        doc = _doc[idx]
        att = atten[idx]
        new = stat(att, doc, topk)[0][0]
        max_right.append( new == pid[idx] )
    max_right = np.array(max_right)
    return max_right

def merge_prediction(attention_score, document, thres=0.18, topk=20):
    predict = []
    for atten, doc in zip(attention_score, document):
        info = stat(atten, doc, topk)
        scr = info[0][2]
        if scr >= thres:
            predict.append(info[0][0])
        else:
            max_appear = max( [_[1] for _ in info] )
            for _ in info:
                if _[1] == max_appear:
                    predict.append( _[0])
                    break
    return np.array(predict).astype(np.int) 


def test_on(_iter, M, sess, fix_attention=False, pure=False):

    running_acc = 0.0
    running_loss = 0.0
    
    point_acc = 0.0
    common_acc = 0.0
    both_acc = 0.0
    merge_acc = 0.0
    # max_acc = 0.0

    counter = 0

    if pure:
        fetch = [M.accuracy, M.loss, M.score]
    else:
        fetch = [M.accuracy, M.loss, M.attention, M.score]

    # sys.stdout.write('\r%d' % counter)

    while True:
        counter += 1
        sys.stdout.write('\r%d' % counter)
        try:
            data = _iter.next()
            if pure:
                accuracy, loss, score = step(data, M, sess, fetch)
            else:
                accuracy, loss, attention, score = step(data, M, sess, fetch)
                if fix_attention:
                    attention = attention[:, :, 0]

                doc = data[1]
                answer = data[-1]
                point, common, both = analyse(doc, answer, attention)
                # max_confirm = extract_and_compare(doc, score, attention)
                merge = merge_prediction(attention, doc, thres=0.186, topk=10)

                point_acc += point.mean()
                common_acc += common.mean()
                both_acc   += both.mean()
                
                merge_acc += (merge == answer.argmax(1)).mean()

            running_loss += loss.mean()
            running_acc += accuracy

        except StopIteration:
            print
            print 'Overall Loss: %.8f, Overall Accuracy: %.8f' % \
                (running_loss / counter, running_acc / counter)

            if not pure:
                print 'AttenAcc Point: %.8f, Common: %.8f, Both: %.8f' \
                            % (point_acc/counter, common_acc/counter, both_acc/counter)
                print '  Merge Acc Rate %.8f' % (merge_acc/counter)

            break


def main(FLAGS):
    # l = define_gpu(1)
    # print 'Using GPU %d' % l[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

    # get data
    test_path = os.path.join(
        FLAGS.data_dir, FLAGS.dataset, 'questions', 'test')
    validate_path = os.path.join(
        FLAGS.data_dir, FLAGS.dataset, 'questions', 'validation')
    test_files = glob(os.path.join(test_path, "*.question"))
    validate_files = glob(os.path.join(validate_path, "*.question"))
    if FLAGS.data_size < len(validate_files):
        validate_files = validate_files[:FLAGS.data_size]
    if FLAGS.data_size < len(test_files):
        test_files = test_files[:FLAGS.data_size]

    # get vocab
    vocab_path = os.path.join(
        FLAGS.data_dir, FLAGS.dataset, '%s.vocab%d' % (FLAGS.dataset, vocab_size))
    with open(vocab_path, 'r') as f:
        vocab = pickle.load(f)

    if FLAGS.pure:
        TensorName.remove('attention')


    # eval
    with tf.Session() as sess:

        if os.path.isdir(FLAGS.load_path):
            ckfiles = choose_ckpt(os.path.join(FLAGS.load_path, 'ckpts'))
            # assert fname is not None
        else:
            ckfiles = [FLAGS.load_path]

        meta = ckfiles[0] + '.meta'
        saver = tf.train.import_meta_graph(meta)
        print 'Graph imported'

        for idx, fname in enumerate(ckfiles):
            print '%d -- %s' % (idx, fname)
            saver.restore(sess, fname)

            M = dig_tensors(sess.graph, targ=TensorName)

            # test dataset
            test_iter = eval_iter(test_files, max_nsteps,
                                  max_query_length, batch_size, vocab)
            test_step = test_iter.next()
            print 'Running on Test data'
            test_on(test_iter, M, sess, pure=FLAGS.pure)

            # validate dataset
            validate_iter = eval_iter(
                validate_files, max_nsteps, max_query_length, batch_size, vocab)
            validate_step = validate_iter.next()
            print 'Running on Validate data'
            test_on(validate_iter, M, sess, pure=FLAGS.pure)


if __name__ == '__main__':

    flags = tf.app.flags
    flags.DEFINE_integer("data_size", None, "")
    flags.DEFINE_integer("gpu", 0, "")

    flags.DEFINE_string("data_dir", "data",
                        "The name of data directory [data]")
    flags.DEFINE_string("dataset", "cnn", "The name of dataset")
    flags.DEFINE_string("load_path", None, "The path to old model. [None]")

    flags.DEFINE_boolean("pure", True, "")
    FLAGS = flags.FLAGS

    if os.path.isdir(FLAGS.load_path):
        with open(os.path.join(FLAGS.load_path, 'Flags.js'), 'r') as f:
            old_flag = json.load(f)
    else:
        js_path = os.path.join(FLAGS.load_path, '../../Flags.js')
        js_path = os.path.abspath(js_path)
        with open(js_path, 'r') as f:
            old_flag = json.load(f)

    max_nsteps = 1000
    max_query_length = 20
    batch_size = old_flag['batch_size']
    vocab_size = old_flag['vocab_size']
    hidden_size = old_flag['hidden_size']
    activation = old_flag['activation']
    attention = old_flag.get('attention', 'concat')
    bidirection = old_flag.get("bidirect", True)
    D = old_flag.get("D", 25)

    main(FLAGS)
