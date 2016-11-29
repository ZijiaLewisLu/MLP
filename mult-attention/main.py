#! /usr/bin/python
import os, sys
import tensorflow as tf
import time
import json
import numpy as np
from utils import pp, define_gpu
from utils import batchIter
from utils.mdu import _load
from model import ML_Attention

flags = tf.app.flags
flags.DEFINE_integer("epoch", 3, "Epoch to train")
flags.DEFINE_integer("vocab_size", 50000, "The size of vocabulary")
flags.DEFINE_integer("batch_size", 32, "The size of batch images")
flags.DEFINE_integer("gpu", 3, "the number of gpus to use")
flags.DEFINE_integer("data_size", None, "Number of files to train on")
flags.DEFINE_integer("hidden_size", 256, "Hidden dimension for rnn and fully connected layer")
flags.DEFINE_integer("embed_size", 128, "Embed size")
flags.DEFINE_float("eval_every", 10.0, "Eval every step")
flags.DEFINE_float("learning_rate", 3e-5, "Learning rate")
# flags.DEFINE_float("momentum", 0.9, "Momentum of RMSProp [0.9]")
# flags.DEFINE_float("decay", 0.95, "Decay of RMSProp [0.95]")
# flags.DEFINE_float("dropout", 0.9, "Dropout rate")
# flags.DEFINE_float("l2_rate", 5e-4, "Dropout rate")
flags.DEFINE_string("log_dir", "log", "Directory name to save the log [log]")
flags.DEFINE_string("data_dir", "data/squad", "Data")
flags.DEFINE_string("load_path", None, "The path to old model. [None]")
# flags.DEFINE_string("optim", 'RMS', "The optimizer to use [RMS]")
FLAGS = flags.FLAGS

# unoften changed parameter
sN=10
sL=50
qL=15
split_rate=0.9
stop_id=2

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.gpu is not None:
        # gpu_list  = define_gpu(FLAGS.gpu)
        # print('  Using GPU:%s' % gpu_list)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

    log_dir = "%s/%s"%(FLAGS.log_dir, time.strftime("%m_%d_%H_%M"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        with open(log_dir+'/Flags.js','w') as f:
            json.dump(FLAGS.__flags, f, indent=4)
    else:
        print('log_dir exist %s' % log_dir)
        exit(2)

    sess = tf.Session()
    model = ML_Attention( FLAGS.batch_size, sN, sL, qL, FLAGS.vocab_size, FLAGS.embed_size, FLAGS.hidden_size, 
                            learning_rate=FLAGS.learning_rate)
    print '  Model Built'

    counter = 0
    start_time = time.time()
    sess.run(tf.initialize_all_variables())
    print '  Variable inited'
    
    train_ids_path = os.path.join(FLAGS.data_dir, 'ids_vocab%d_train.txt' % FLAGS.vocab_size)
    data = _load(train_ids_path)
    
    if FLAGS.data_size:
        data = data[:FLAGS.data_size]
    part = int(np.floor(len(data) * split_rate))
    train = data[:part]
    # validate = data[part:]
    print '  Data Loaded, Start to train'

    for epoch_idx in range(FLAGS.epoch):
        np.random.shuffle(train)
        titer = batchIter(FLAGS.batch_size, train, sN, sL, qL, stop_id=stop_id)
        tstep = titer.next()
        running_acc = 0.0
        running_loss = 0.0 

        feed_in = [
            model.loss,
            model.accuracy,
            model.train_op,
            model.check_op,
            # model.gvs
        ]
        for batch_idx, P, p_len, Q, q_len, A in titer:

            loss, accuracy, _, _ = sess.run( feed_in, 
                                            feed_dict={
                                                model.passage: P,
                                                model.p_len: p_len,
                                                model.query: Q,
                                                model.q_len: q_len,
                                                model.answer: A
                                            } )
            running_acc += accuracy
            running_loss += np.mean(loss)
            if counter % FLAGS.eval_every == 0:
                print "Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f" \
                    % (epoch_idx, batch_idx, tstep, time.time() - start_time, running_loss/FLAGS.eval_every, running_acc/FLAGS.eval_every)
                sys.stdout.flush()
                running_loss = 0.0
                running_acc = 0.0

                # vs = sess.run( variable )
                # try:
                #     gs = sess.run( gradient )
                # except Exception, e:
                #     print e
                #     import ipdb; ipdb.set_trace()
                # print '======================================='
                # for name, ( gra, val ) in zip( model.names, gvs ):

                #     if not isinstance(gra, np.ndarray):
                #         gra = gra.values

                #     print name
                #     print 'V> ', np.mean(val), np.max(val), np.min(val)
                #     print 'G> ', np.mean(gra), np.max(gra), np.min(gra)

                #     if not np.isfinite(val).all() or not np.isfinite(gra).all():
                #         import ipdb; ipdb.set_trace()
                

                # print '======================================='
                # print ''


            counter += 1

            # if (counter+1) % FLAGS.eval_every == 0:
                # np.random.shuffle(validate)

          

if __name__ == '__main__':
    try:
        tf.app.run()
    except Exception, e:
        print e
        import ipdb; ipdb.set_trace()
  