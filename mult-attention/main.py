#! /usr/bin/python
import os
import tensorflow as tf
import time
import json
import numpy as np
from utils import pp, define_gpu
from utils import load_data
from model import ML_Attention

flags = tf.app.flags
flags.DEFINE_integer("epoch", 15, "Epoch to train [40]")
flags.DEFINE_integer("vocab_size", 50000, "The size of vocabulary [10000]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
flags.DEFINE_integer("gpu", 1, "the number of gpus to use")
flags.DEFINE_integer("data_size", None, "Number of files to train on")
flags.DEFINE_integer("hidden_size", 128, "Hidden dimension for rnn and fully connected layer")
flags.DEFINE_integer("embed_size", 128, "Embed size")
flags.DEFINE_integer("eval_every", 1000, "Eval every step")
flags.DEFINE_float("learning_rate", 3e-5, "Learning rate [0.00005]")
# flags.DEFINE_float("momentum", 0.9, "Momentum of RMSProp [0.9]")
# flags.DEFINE_float("decay", 0.95, "Decay of RMSProp [0.95]")
# flags.DEFINE_float("dropout", 0.9, "Dropout rate")
# flags.DEFINE_float("l2_rate", 5e-4, "Dropout rate")
flags.DEFINE_string("log_dir", "log", "Directory name to save the log [log]")
flags.DEFINE_string("data_dir", "data/squad", "Data")
flags.DEFINE_string("load_path", None, "The path to old model. [None]")
flags.DEFINE_string("optim", 'RMS', "The optimizer to use [RMS]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.gpu is not None:
        gpu_list  = define_gpu(FLAGS.gpu)

    log_dir = "%s/%s"%(FLAGS.log_dir, time.strftime("%m_%d_%H_%M"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        with open(log_dir+'/Flags.js','w') as f:
            json.dump(FLAGS.__flags, f, indent=4)
    else:
        print('log_dir exist %s' % log_dir)
        exit(2)

    sess = tf.Session()
    model = ML_Attention( FLAGS.batch_size, 10, 50, 15, FLAGS.vocab_size, FLAGS.embed_size, FLAGS.hidden_size, 
                            learning_rate=FLAGS.learning_rate)
    print 'Model Built'

    counter = 0
    start_time = time.time()
    sess.run(tf.initialize_all_variables())
    print 'Variable inited'
    print 'Start to train'
    for epoch_idx in range(FLAGS.epoch):
        titer, tstep, viter, vstep = load_data(data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size, 
                                                vocab_size=FLAGS.vocab_size,
                                                data_size=FLAGS.data_size, sN=10, sL=50, qL=15)
        print 'Data Loaded'

        running_acc = 0.0
        running_loss = 0.0 
        for batch_idx, P, p_len, Q, q_len, A in titer:
            _, loss, accuracy = sess.run( [model.train_op, model.loss, model.accuracy], 
                                            feed_dict={
                                                model.passage: P,
                                                model.p_len: p_len,
                                                model.query: Q,
                                                model.q_len: q_len,
                                                model.answer: A
                                            } )
            running_acc += accuracy
            running_loss += np.mean(loss)
            if counter % 10 == 0:
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f"
                    % (epoch_idx, batch_idx, tstep, time.time() - start_time, running_loss/10.0, running_acc/10.0))
                running_loss = 0.0
                running_acc = 0.0
            counter += 1

          

if __name__ == '__main__':
  tf.app.run()