#! /usr/bin/python
import os
import tensorflow as tf
from glob import glob
import pickle
from utils import data_to_token_ids, define_gpu
import json
import numpy as np
from attentive_model import AttentiveReader

flags = tf.app.flags
flags.DEFINE_integer("data_size", None, "")
flags.DEFINE_string("data_dir", "data", "The name of data directory [data]")
flags.DEFINE_string("dataset", "cnn", "The name of dataset [cnn, dailymail]")
flags.DEFINE_string("load_path", None, "The path to old model. [None]")
FLAGS = flags.FLAGS


if os.path.isdir(FLAGS.load_path):
    with open(os.path.join(FLAGS.load_path, 'Flags.js'), 'r') as f:
        old_flag = json.load(f)
else:
    js_path = os.path.join(FLAGS.load_path, '../../Flags.js')
    js_path = os.path.abspath(js_path)
    with open(js_path, 'r') as f:
        old_flag = json.load(f)

max_nsteps=1000
max_query_length=20
batch_size = old_flag['batch_size']
vocab_size = old_flag['vocab_size']
hidden_size = old_flag['hidden_size']
activation = old_flag['activation']


    
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
            rslt  =  data_to_token_ids(fname, None, vocab, save=False)
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


def run(sess, model, data_iter, steps):
    running_loss = 0.0
    running_acc = 0.0
    for batch_idx, docs, d_end, queries, q_end, y in data_iter:
        feed={model.document: docs,
                   model.query: queries,
                   model.d_end: d_end,
                   model.q_end: q_end,
                   model.y: y, 
                   model.dropout: 1.0,
                   }
        cost, accuracy = sess.run([model.loss, model.accuracy], feed)
        running_acc += accuracy
        running_loss += np.mean(cost)
        if batch_idx % 5 == 0:
            print "[%4d/%4d] loss: %.8f, accuracy: %.8f" \
                        % ( batch_idx, steps, running_loss/float(batch_idx+1), running_acc/float(batch_idx+1))
            
    print 'Overall Loss: %.8f, Overall Accuracy: %.8f' % \
        ( running_loss/float(batch_idx+1), running_acc/float(batch_idx+1) )

def main(_):
    l = define_gpu(1)
    print 'Using GPU %d' % l[0]

    # get data
    test_path = os.path.join( FLAGS.data_dir, FLAGS.dataset, 'questions', 'test' )
    validate_path = os.path.join( FLAGS.data_dir, FLAGS.dataset, 'questions', 'validation' )
    test_files = glob(os.path.join(test_path, "*.question"))
    validate_files = glob(os.path.join(validate_path, "*.question"))
    if FLAGS.data_size < len(validate_files):
        validate_files = validate_files[:FLAGS.data_size]
    if FLAGS.data_size < len(test_files):
        test_files = test_files[:FLAGS.data_size]
     
    # get vocab   
    vocab_path = os.path.join( FLAGS.data_dir, FLAGS.dataset, '%s.vocab%d'%(FLAGS.dataset, vocab_size) )
    with open(vocab_path, 'r') as f:
        vocab = pickle.load(f)

    # eval
    with tf.Session() as sess:
        model = AttentiveReader(batch_size=batch_size, size=hidden_size, activation=activation)
        model.prepare_model()
        print "Model Build"

        saver = tf.train.Saver()
        if os.path.isdir(FLAGS.load_path):
            fname = tf.train.latest_checkpoint(os.path.join(FLAGS.load_path, 'ckpts'))
            assert fname is not None 
        else:
            fname = FLAGS.load_path
        print "Loading %s"% fname
        saver.restore(sess, fname)

        # test dataset
        test_iter = eval_iter( test_files, max_nsteps, max_query_length, batch_size, vocab )
        test_step = test_iter.next()
        print 'Running on Test data'
        run(sess, model, test_iter, test_step)
                
        # validate dataset
        validate_iter = eval_iter( validate_files, max_nsteps, max_query_length, batch_size, vocab )
        validate_step = validate_iter.next()
        print 'Running on Validate data'
        run(sess, model, validate_iter, validate_step)

if __name__ == '__main__':
    main(0)
