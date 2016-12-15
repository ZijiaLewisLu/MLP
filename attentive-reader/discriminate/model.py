import tensorflow as tf
from utils import load_dataset
import numpy as np
import h5py
import os

VFILE = None
TFILE = None


class Discriminator(object):

    def __init__(self, batch_size, hidden_size,
                 learning_rate=1e-4,
                 num_layer=2,
                 sequence_length=1000,
                 reuse=True):

        self.data = tf.placeholder(
            tf.float32, [batch_size, sequence_length, 2], 'data')
        self.d_len = tf.placeholder(tf.int32, [batch_size], 'd_len')
        self.label = tf.placeholder(tf.int32, [batch_size], 'label')
        self.global_step = tf.Variable(1, name='global_step', trainable=False)


        if reuse:
            self.cells = [tf.nn.rnn_cell.LSTMCell(
                hidden_size, state_is_tuple=True)] * num_layer
        else:
            self.cells = [tf.nn.rnn_cell.LSTMCell(
                hidden_size, state_is_tuple=True) for _ in range(num_layer)]

        cell = tf.nn.rnn_cell.MultiRNNCell(self.cells, state_is_tuple=True)

        hidden, last_state = tf.nn.dynamic_rnn(
            cell,
            self.data,
            sequence_length=self.d_len,
            dtype=tf.float32
        )

        W = tf.get_variable('W', [batch_size], dtype=tf.float32)
        self.score = tf.mulmat(hidden, W, name='score')
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
            self.score, self.label, name='loss')

        correct = tf.to_int32(self.score == self.label)
        self.accuracy = tf.reduce_mean(correct, name='accuracy')

        # optim = tf.train.GradientDescentOptimizer(learning_rate)
        self.optim = tf.train.AdamOptimizer(learning_rate)

        self.train_op = self.optim.minimize(self.loss, global_step=self.global_step)

def step(sess, model, data, fetch):
    idx, d, dl, l = data
    # prin
    return sess.run(fetch,
                    feed_dict={
                        model.data: d,
                        model.d_len: dl,
                        model.label: l,
                    })


def data_iter(batch_size, data, d_len, label, shuffle_data=True):
    N = data.shape[0]

    steps = np.ceil(N / float(batch_size))
    steps = int(steps)
    yield steps

    for s in range(steps):
        head = s * batch_size
        end = (s + 1) * batch_size
        if end <= N:
            d = data[head:end]
            l = label[head:end]
        else:
            d = data[head:] + data[:end - N]
            l = label[head:] + label[:end - N]

        if shuffle_data:
            np.random.shuffle(d)
            np.random.shuffle(l)

        yield s, d, l


def prepare_data(batch_size, fname, vocab_size=50003, shuffle=True):

    titer, tstep, viter, vstep = load_dataset('data', 'cnn',
                                              vocab_size, batch_size, 1000, 20, shuffle_data=False)

    with h5py.File(fname, 'r') as hf:
        data = hf.get('attention_score')
        attention = np.array(data)
        del data

    print attention.shape

    itr = data_iter(batch_size, attention, shuffle_data=shuffle)
    step = itr.next()
    return itr, step


def create_flag():
    flags = tf.app.flags
    flags.DEFINE_integer("epoch", 15, "Epoch to train [40]")
    flags.DEFINE_integer("batch_size", 128, "The size of batch images [32]")
    flags.DEFINE_integer("gpu", 2, "the number of gpus to use")
    flags.DEFINE_integer("data_size", None, "Number of files to train on")
    flags.DEFINE_integer("hidden_size", 256,
                         "Hidden dimension for rnn and fully connected layer")
    flags.DEFINE_integer("eval_every", 1000, "Eval every step")
    flags.DEFINE_integer("layer", 2, "Eval every step")
    flags.DEFINE_float("learning_rate", 5e-5, "Learning rate [0.00005]")
    flags.DEFINE_string(
        "log_dir", "log", "Directory name to save the log [log]")
    flags.DEFINE_string("load_path", None, "The path to old model. [None]")
    flags.DEFINE_string("optim", 'RMS', "The optimizer to use [RMS]")
    flags.DEFINE_boolean('reuse', True, '')

    FLAGS = flags.FLAGS
    return FLAGS


def main():
    FLAGS = create_flag()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

    M = Discriminator( 
            FLAGS.batch_size, FLAGS.hidden_size,
            learning_rate=FLAGS.learning_rate,
            num_layer=FLAGS.layer,
            reuse=FLAGS.reuse,
            )

    # titer, tstep = prepare_data(FLAGS.batch_size, TFILE)
    # viter, vstep = prepare_data(FLAGS.batch_size, VFILE)

    with tf.Session() as sess:
        fetch = [ M.global_step, M.loss, M.accuracy, M.train_op ]
        vfetch =[ M.loss, M.accuracy ] 

        sess.run(tf.initialize_all_variables())

        running_acc = 0.0
        running_loss = 0.0
        for e in FLAGS.epoch:
            titer, tstep = prepare_data(FLAGS.batch_size, TFILE, shuffle=True)

            for data in titer:
                # print data[0]
                gstep, loss, accuracy, _ = step(sess, M, data, fetch)
                running_acc += accuracy
                running_loss+= loss.mean()

                if gstep % 20 == 0:
                    print '%d E[%d] Acc: %.4f Loss: %.4f' % \
                        (gstep, e, running_acc/20, running_loss/20)
                    running_acc = 0.0
                    running_loss = 0.0

                if gstep % FLAGS.eval_every == 0:

                    viter, vstep = prepare_data(FLAGS.batch_size, VFILE, shuffle=False)
                    vrunning_acc = 0.0
                    vrunning_loss = 0.0
                    
                    for data in viter:
                        # print data[0]
                        loss, accuracy = step(sess, M, data, vfetch)
                        vrunning_acc += accuracy
                        vrunning_loss+= loss.mean()

                    print 'Evaluate Acc: %.4f Loss: %.4f' % \
                                    (vrunning_acc/vstep, vrunning_loss/vstep)

if __name__ == '__main__':
    main()
