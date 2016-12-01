#! /usr/bin/python
import os
import sys
import tensorflow as tf
import time
import json
import numpy as np
from utils import pp, define_gpu
from mdu import batchIter
from mdu import _load
from model import ML_Attention

flags = tf.app.flags
flags.DEFINE_integer("epoch", 60, "Epoch to train")
flags.DEFINE_integer("vocab_size", 50000, "The size of vocabulary")
flags.DEFINE_integer("batch_size", 32, "The size of batch images")
flags.DEFINE_integer("gpu", 3, "the number of gpus to use")
flags.DEFINE_integer("data_size", None, "Number of files to train on")
flags.DEFINE_integer("hidden_size", 256, "Hidden dimension for rnn and fully connected layer")
flags.DEFINE_integer("embed_size", 256, "Embed size")
flags.DEFINE_float("eval_every", 80.0, "Eval every step")
flags.DEFINE_float("save_every", 500.0, "Eval every step")
flags.DEFINE_float("learning_rate", 3e-5, "Learning rate")
# flags.DEFINE_float("momentum", 0.9, "Momentum of RMSProp [0.9]")
# flags.DEFINE_float("decay", 0.95, "Decay of RMSProp [0.95]")
# flags.DEFINE_float("dropout", 0.9, "Dropout rate")
flags.DEFINE_float("l2_rate", 0.0, "Dropout rate")
flags.DEFINE_string("log_dir", "log", "Directory name to save the log [log]")
flags.DEFINE_string("data_dir", "data/squad", "Data")
flags.DEFINE_string("load_path", None, "The path to old model.")
flags.DEFINE_string("optim", 'SGD', "The optimizer to use")
FLAGS = flags.FLAGS

# unoften changed parameter
sN=10
sL=50
qL=15
stop_id=2
val_rate = 0.05

def initialize(sess, saver, load_path=None):
    if not load_path:
        sess.run(tf.initialize_all_variables())
    else:
        if os.path.isdir(load_path):
            fname = tf.train.latest_checkpoint(os.path.join(load_path, 'ckpts'))
            assert fname is not None
        else:
            fname = load_path
        print "  Load from %s" % fname
        saver.restore(sess, fname)

def prepare_data(path, data_size=None, size=3185, val_rate=0.05):
    train_data = _load(path)
    validate_data = train_data[-size:]
    train_data = train_data[:-size]

    if data_size:
        train_data=train_data[:data_size]
    vsize = max( 20, len(train_data)*val_rate)
    vsize = int( min(vsize, len(validate_data)) )
    return train_data, validate_data, vsize

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.gpu is not None:
        # gpu_list  = define_gpu(FLAGS.gpu)
        # print('  Using GPU:%s' % gpu_list)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

    sess = tf.Session()
    model = ML_Attention( FLAGS.batch_size, sN, sL, qL, FLAGS.vocab_size, FLAGS.embed_size, FLAGS.hidden_size, 
                            learning_rate=FLAGS.learning_rate,
                            l2_rate=FLAGS.l2_rate,
                            optim=FLAGS.optim)
    print '  Model Built'

    saver = tf.train.Saver()
    initialize(sess, saver, FLAGS.load_path)
    print '  Variable inited'
    
    ids_path = os.path.join(FLAGS.data_dir, 'ids_vocab%d_train.txt' % FLAGS.vocab_size)
    train_data, validate_data, vsize = prepare_data(ids_path, data_size=FLAGS.data_size, val_rate=val_rate)
    print '  Data Loaded'

    log_dir = "%s/%s"%(FLAGS.log_dir, time.strftime("%m_%d_%H_%M"))
    save_dir = os.path.join(log_dir, 'ckpts')
    if os.path.exists(save_dir):
        print('log_dir exist %s' % log_dir)
        exit(2)
    os.makedirs(save_dir)
    with open(log_dir+'/Flags.js','w') as f:
        json.dump(FLAGS.__flags, f, indent=4)
    print '  Writing log to %s' % log_dir

    counter = 1
    vcounter = 1
    start_time = time.time()
    writer = tf.train.SummaryWriter(log_dir, sess.graph)

    for epoch_idx in range(FLAGS.epoch):
        np.random.shuffle(train_data)
        titer = batchIter(FLAGS.batch_size, train_data, sN, sL, qL, stop_id=stop_id)
        tstep = titer.next()
        running_acc = 0.0
        running_loss = 0.0 

        feed_in = [
            model.loss,
            model.accuracy,
            model.train_op,
            model.check_op,
            model.train_summary,
            # model.gvs
        ]
        for batch_idx, P, p_len, Q, q_len, A in titer:

            loss, accuracy, _, _, sum_str = sess.run( feed_in, 
                                            feed_dict={
                                                model.passage: P,
                                                model.p_len: p_len,
                                                model.query: Q,
                                                model.q_len: q_len,
                                                model.answer: A
                                            } )
            running_acc += accuracy
            running_loss += np.mean(loss)
            writer.add_summary( sum_str, counter )

            if counter % 20 == 0:
                print "Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f" \
                    % (epoch_idx, batch_idx, tstep, time.time() - start_time, running_loss/20.0, running_acc/20.0)
                sys.stdout.flush()
                running_loss = 0.0
                running_acc = 0.0

            if counter % FLAGS.eval_every == 0:
                _accuracy = 0.0
                _loss = 0.0
                idxs = np.random.choice(len(validate_data), size=vsize)
                D = [ validate_data[idx] for idx in idxs ]
                viter = batchIter(FLAGS.batch_size, D, sN, sL, qL, stop_id=stop_id)
                vstep = float(viter.next())
                for batch_idx, P, p_len, Q, q_len, A in viter:
                    loss, accuracy, sum_str = sess.run( [model.Vloss, model.Vaccuracy, model.validate_summary],
                                            feed_dict={
                                                model.passage: P,
                                                model.p_len: p_len,
                                                model.query: Q,
                                                model.q_len: q_len,
                                                model.answer: A
                                            } )
                    _accuracy += accuracy
                    _loss += np.mean(loss)
                    vcounter += 1
                    writer.add_summary( sum_str, vcounter )
                print '  Evaluation: time: %4.4f, loss: %.8f, accuracy: %.8f' % \
                                (time.time()-start_time, _loss/vstep, _accuracy/vstep)
                vcounter += int(vstep/4.0)

            if counter % FLAGS.save_every == 0:
                fname = os.path.join(save_dir, 'model')
                print "  Saving Model..."
                saver.save(sess, fname, global_step=counter)

            counter += 1
         
if __name__ == '__main__':
    tf.app.run()
    # try:
    #     tf.app.run()
    # except Exception, e:
    #     import ipdb, traceback
    #     etype, value, tb = sys.exc_info()
    #     traceback.print_exc()
    #     ipdb.post_mortem(tb)
