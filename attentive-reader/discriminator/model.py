# /usr/bin/python
import tensorflow as tf
import numpy as np
import h5py
import os

VFILE = './Data.h5'
TFILE = './Validate.h5'


class Discriminator(object):

    def __init__(self, batch_size, hidden_size,
                 learning_rate=1e-4,
                 num_layer=2,
                 sequence_length=1000,
                 topk=True,
                 reuse=True):

        self.data = tf.placeholder(
            tf.float32, [batch_size, sequence_length, 2], 'data')
        self.d_len = tf.placeholder(tf.int32, [batch_size], 'd_len')
        self.label = tf.placeholder(tf.int32, [batch_size, 3], 'label')
        self.global_step = tf.Variable(1, name='global_step', trainable=False)

        if reuse:
            self.cells = [tf.nn.rnn_cell.LSTMCell(
                hidden_size, state_is_tuple=True)] * num_layer
        else:
            self.cells = [tf.nn.rnn_cell.LSTMCell(
                hidden_size, state_is_tuple=True) for _ in range(num_layer)]

        cell = tf.nn.rnn_cell.MultiRNNCell(self.cells, state_is_tuple=True)

        if topk:
            hidden, last_state = tf.nn.rnn(
                cell,
                tf.unpack(self.data, axis=1),
                dtype=tf.float32,
                )

            final = hidden[-1]

        else:
            hidden, last_state = tf.nn.dynamic_rnn(
                cell,
                self.data,
                sequence_length=self.d_len,
                dtype=tf.float32,
            )

            with tf.name_scope('final_state'):
                unpack = tf.unpack(hidden, axis=0)
                begins = tf.pack(
                    [self.d_len - 1, tf.zeros([batch_size], dtype=tf.int32)], axis=1)  # N, 2
                begins = tf.unpack(begins)
                final_state = []
                for h, b in zip(unpack, begins):
                    f = tf.slice(h, b, [1, hidden_size])
                    print f.get_shape()
                    final_state.append(f)

                final = tf.pack(final_state)

                final = tf.squeeze(final_state, [1])


        self.final_sparstity = tf.nn.zero_fraction(
                   final, name='final_hidden_sparisty')     
        final = tf.nn.relu(final, name='relu')

        W = tf.get_variable('W', [hidden_size, 3], dtype=tf.float32)
        self.score = tf.matmul(final, W, name='score')

        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            self.score, self.label, name='loss')

        self.prediction = tf.argmax(self.score, 1, name='prediction')
        self.right_label = tf.argmax(self.label, 1, name='right_label')
        # print self.prediction.get_shape()
        self.correct = tf.equal(
            self.prediction, self.right_label, name='correct_prediction')
        self.accuracy = tf.reduce_mean(
            tf.to_float(self.correct), name='accuracy')

        # optim = tf.train.GradientDescentOptimizer(learning_rate)
        self.optim = tf.train.AdamOptimizer(learning_rate)

        self.train_op = self.optim.minimize(
            self.loss, global_step=self.global_step)


def step(sess, model, data, fetch):
    idx, d, dl, l = data
    # prin
    return sess.run(fetch,
                    feed_dict={
                        model.data: d,
                        model.d_len: dl,
                        model.label: l,
                    })


def Topk(array, k):
    top = np.argpartition(-array, k)[:k]
    pair = sorted(zip(array[top], top), key=lambda x: x[0], reverse=True)
    return [_[1] for _ in pair]


def data_iter(batch_size, data, d_len, label, k=None, shuffle_data=True):
    N = data.shape[0]

    steps = np.ceil(N / float(batch_size))
    steps = int(steps)
    yield steps

    d_topk = np.zeros( [batch_size, k, 2], dtype=np.float32 )
    oh_label = np.zeros([batch_size, 3], dtype=np.int)

    for s in range(steps):

        oh_label.fill(0)
        d_topk.fill(0)

        head = s * batch_size
        end = (s + 1) * batch_size
        if end <= N:
            d = data[head:end]
            dl = d_len[head:end]
            l = label[head:end]
        else:
            d = np.concatenate([data[head:], data[:end - N]], axis=0)
            l = np.concatenate([label[head:], label[:end - N]], axis=0)
            dl = np.concatenate([d_len[head:], d_len[:end - N]], axis=0)

        if shuffle_data:
            np.random.shuffle(d)
            np.random.shuffle(l)
            np.random.shuffle(dl)

        try:
            oh_label[range(batch_size), l - 1] = 1
        except:
            import ipdb
            ipdb.set_trace()

        if k:
            for i in range(batch_size):
                top_idx = np.argpartition(-d[i,:,1], k)[:k]
                top_idx = sorted(top_idx)
                d_topk[i] = d[i,top_idx,:] # i, 2, k

            yield s, d_topk, dl, oh_label
        else:
            yield s, d, dl, oh_label


def prepare_data(batch_size, fname, vocab_size=50003, topk=None, shuffle=True):

    with h5py.File(fname, 'r') as hf:
        data = hf.get('data')
        dlen = hf.get('dlen')
        label = hf.get('label')

        data = np.array(data)
        dlen = np.array(dlen)
        label = np.array(label)

    itr = data_iter(batch_size, data, dlen, label, shuffle_data=shuffle, k=topk)
    step = itr.next()
    return itr, step


def create_flag():
    flags = tf.app.flags
    flags.DEFINE_integer("epoch", 15, "Epoch to train [40]")
    flags.DEFINE_integer("batch_size", 5, "")
    flags.DEFINE_integer("gpu", 2, "the number of gpus to use")
    flags.DEFINE_integer("data_size", None, "Number of files to train on")
    flags.DEFINE_integer("hidden_size", 64, "")
    flags.DEFINE_integer("eval_every", 1000, "Eval every step")
    flags.DEFINE_integer("layer", 2, "Eval every step")
    flags.DEFINE_integer("topk", None, "")

    flags.DEFINE_float("learning_rate", 5e-2, "Learning rate [0.00005]")
    flags.DEFINE_string("log_dir", "log", "")
    flags.DEFINE_string("load_path", None, "The path to old model. [None]")
    flags.DEFINE_string("data_path", 'Test.h5', "")

    flags.DEFINE_string("optim", 'RMS', "The optimizer to use [RMS]")
    flags.DEFINE_boolean('reuse', True, '')

    FLAGS = flags.FLAGS

    # print FLAGS
    for k in FLAGS.__flags:
        print k, FLAGS.__flags[k]

    return FLAGS


def main():
    FLAGS = create_flag()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

    if FLAGS.topk:
        M = Discriminator(
            FLAGS.batch_size, FLAGS.hidden_size,
            learning_rate=FLAGS.learning_rate,
            sequence_length=FLAGS.topk,
            num_layer=FLAGS.layer,
            reuse=FLAGS.reuse,
            topk=True,
        )        
    else:
        M = Discriminator(
            FLAGS.batch_size, FLAGS.hidden_size,
            learning_rate=FLAGS.learning_rate,
            num_layer=FLAGS.layer,
            reuse=FLAGS.reuse,
            topk=False,
        )

    # titer, tstep = prepare_data(FLAGS.batch_size, TFILE)
    # viter, vstep = prepare_data(FLAGS.batch_size, VFILE)

    with tf.Session() as sess:
        fetch = [M.global_step, M.loss, M.accuracy, M.score, M.train_op,
                 M.final_sparstity,
                 # M.prediction, M.right_label,
                 # M.correct

                 ]
        vfetch = [M.loss, M.accuracy]

        sess.run(tf.initialize_all_variables())

        running_acc = 0.0
        running_loss = 0.0
        for e in range(FLAGS.epoch):
            titer, tstep = prepare_data(
                FLAGS.batch_size, FLAGS.data_path, shuffle=True, topk=FLAGS.topk)

            for data in titer:
                # print data[0]
                gstep, loss, accuracy, score, _, spars = step(
                    sess, M, data, fetch)
                running_acc += accuracy
                running_loss += loss.mean()

                # print 'sparse', spars

                if gstep % 20 == 0:
                    print '%d E[%d] Acc: %.4f Loss: %.4f' % \
                        (gstep, e, running_acc / 20.0, running_loss / 20.0)
                    running_acc = 0.0
                    running_loss = 0.0

                # if gstep % FLAGS.eval_every == 0:

                #     viter, vstep = prepare_data(FLAGS.batch_size, VFILE, shuffle=False, topk=FLAGS.topk)
                #     vrunning_acc = 0.0
                #     vrunning_loss = 0.0

                #     for data in viter:
                #         # print data[0]
                #         loss, accuracy = step(sess, M, data, vfetch)
                #         vrunning_acc += accuracy
                #         vrunning_loss+= loss.mean()

                #     print 'Evaluate Acc: %.4f Loss: %.4f' % \
                #                     (vrunning_acc/vstep, vrunning_loss/vstep)

if __name__ == '__main__':
    main()
