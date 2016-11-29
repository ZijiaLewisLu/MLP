import os
import tensorflow as tf
import time
import json
import numpy as np
from attentive_model import AttentiveReader

from utils import pp, define_gpu, MultiGPU_Manager, load_dataset

flags = tf.app.flags
flags.DEFINE_integer("epoch", 15, "Epoch to train")
flags.DEFINE_integer("vocab_size", 50003, "The size of vocabulary")
flags.DEFINE_integer("batch_size", 16, "The size of batch images")
flags.DEFINE_integer("gpu", 2, "the number of gpus to use")
flags.DEFINE_integer("data_size", 3000, "Number of files to train on")
flags.DEFINE_integer("hidden_size", 128,
                     "Hidden dimension for rnn and fully connected layer")
flags.DEFINE_float("learning_rate", 5e-5, "Learning rate [0.00005]")
flags.DEFINE_float("momentum", 0.9, "Momentum of RMSProp [0.9]")
flags.DEFINE_float("decay", 0.95, "Decay of RMSProp [0.95]")
flags.DEFINE_float("dropout", 1.0, "Dropout rate")
flags.DEFINE_string("data_dir", "data", "The name of data directory [data]")
flags.DEFINE_string("dataset", "cnn", "The name of dataset [cnn, dailymail]")
flags.DEFINE_string("log_dir", "log", "Directory name to save the log [log]")
flags.DEFINE_string("load_path", None, "The path to old model. [None]")
flags.DEFINE_string("optim", 'RMS', "The optimizer to use [RMS]")
flags.DEFINE_string("activation", 'none',
                    "The the last activation layer to use before Softmax loss")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    # create log dir
    log_dir = "%s/%s" % (FLAGS.log_dir,
                         time.strftime("%m_%d_%H_%M"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        with open(log_dir + '/Flags.js', 'w') as f:
            json.dump(FLAGS.__flags, f, indent=4)
    else:
        print('log_dir exist %s' % log_dir)
        exit(2)

    # create graph
    def model_gen():
        m = AttentiveReader(batch_size=FLAGS.batch_size,
                            dropout_rate=FLAGS.dropout,
                            momentum=FLAGS.momentum,
                            decay=FLAGS.decay,
                            size=FLAGS.hidden_size,
                            use_optimizer=FLAGS.optim,
                            activation=FLAGS.activation)

        m.prepare_model(parallel=True)
        return m

    gpu_list = define_gpu(FLAGS.gpu)
    print " [*] Using GPU: ", gpu_list

    print "\n\n [*] Building Network..."
    start = time.time()
    mgr = MultiGPU_Manager(range(FLAGS.gpu), model_gen)
    print " [*] Preparing model finished. Use %4.4f" % (time.time() - start)

    print " [*] Initialize Variables..."
    start = time.time()
    if FLAGS.load_path:
        mgr.init_variable(load_path=os.path.join(FLAGS.load_path, 'ckpts'))
    else:
        mgr.init_variable()
    if mgr.load_path is not None:
        print " [*] Load from %s" % mgr.load_path
    print " [*] Variables inited. Use %4.4f" % (time.time() - start)

    # Summary
    writer = tf.train.SummaryWriter(log_dir, mgr.sess.graph)
    print " [*] Writing log to %s" % log_dir

    # setup parameter
    sess = mgr.sess
    M = mgr.main_model
    max_nsteps = M.max_nsteps
    max_query_length = M.max_query_length
    total_batch_size = mgr.N * FLAGS.batch_size
    start_time = time.time()
    # ACC = []
    # LOSS = []
    train_ops = [mgr.train_op, M.train_sum, mgr.loss, mgr.accuracy]
    validate_ops = [M.validate_sum, mgr.loss, mgr.accuracy]
    counter = 0

    for epoch_idx in xrange(FLAGS.epoch):
        # load data
        train_iter, tsteps, validate_iter, vsteps = load_dataset(FLAGS.data_dir, FLAGS.dataset, FLAGS.vocab_size,
                                                                 total_batch_size, max_nsteps, max_query_length,
                                                                 size=FLAGS.data_size)

        # train
        for batch_idx, docs, d_end, queries, q_end, y in train_iter:
            feed = mgr.feed_dict({'document': docs, 'query': queries,
                                  'd_end': d_end, 'q_end': q_end,
                                  'y': y})
            _, summary_str, cost, accuracy = sess.run(train_ops, feed)

            writer.add_summary(summary_str, counter)
            if counter % 10 == 0:
                print "Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f" \
                      % (epoch_idx, batch_idx, tsteps, time.time() - start_time, np.mean(cost), accuracy)
            counter += 1

        # validate
        running_acc = 0
        running_loss = 0
        for batch_idx, docs, d_end, queries, q_end, y in validate_iter:
            feed = mgr.feed_dict({'document': docs, 'query': queries,
                                  'd_end': d_end, 'q_end': q_end,
                                  'y': y})
            validate_sum_str, cost, accuracy = sess.run(validate_ops, feed)
            writer.add_summary(validate_sum_str, counter)
            running_acc += accuracy
            running_loss += np.mean(cost)

        # ACC.append(running_acc/vsteps)
        # LOSS.append(running_loss/vsteps)
        print "Epoch: [%2d] Validation time: %4.4f, loss: %.8f, accuracy: %.8f" \
              % (epoch_idx, time.time() - start_time, running_loss / vsteps, running_acc / vsteps)

        # save
        if (epoch_idx + 1) % 3 == 0:
            print " [*] Saving checkpoints..."
            checkpoint_dir = os.path.join(log_dir, "ckpts")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            fname = os.path.join(checkpoint_dir, 'model')
            mgr.save(fname, global_step=counter)
        print '\n\n'


if __name__ == '__main__':
    tf.app.run()
