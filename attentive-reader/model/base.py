import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell

import sys
sys.path.insert(0, '..')
from utils import fetch_files, data_iter #apply_attention
from utils.attention import local_attention

def norm(x):
    if not isinstance(x, np.ndarray):
        x = x.values
    return np.sqrt((x**2).sum())


class BaseModel(object):
    """Attentive Reader."""

    def __init__(self, vocab_size=50003, batch_size=32,
                 learning_rate=1e-4, momentum=0.9, decay=0.95, l2_rate=1e-4,
                 size=256,
                 max_nsteps=1000,
                 max_query_length=20,
                 use_optimizer='RMS',
                 activation='tanh',
                 attention='bilinear',
                 bidirection=True,
                 D=5,
                 max_norm=6,
                 ):

        self.size = size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_nsteps = max_nsteps
        self.max_query_length = max_query_length
        self.vocab_size = vocab_size
        # self.dropout = dropout_rate
        self.l2_rate = l2_rate
        self.momentum = momentum
        self.decay = decay
        self.use_optimizer = use_optimizer
        self.activation = activation
        self.attention = attention
        self.bidirection = bidirection
        self.D = D
        self.max_norm=max_norm

        self.saver = None

    def construct_inputs(self):
        self.document = tf.placeholder(
            tf.int32, [self.batch_size, self.max_nsteps], name='document')
        self.query = tf.placeholder(
            tf.int32, [self.batch_size, self.max_query_length], name='query')
        self.d_end = tf.placeholder(tf.int32, self.batch_size, name='docu-end')
        self.q_end = tf.placeholder(tf.int32, self.batch_size, name='quer-end')
        self.y = tf.placeholder(
            tf.float32, [self.batch_size, self.vocab_size], name='Y')
        self.dropout = tf.placeholder(tf.float32, name='dropout_rate')

    def construct_loss_and_summary(self, score, parallel=False):

        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            score, self.y, name='loss')
        loss_sum = tf.scalar_summary("T_loss", tf.reduce_mean(self.loss))

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(score, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, "float"), name='accuracy')
        acc_sum = tf.scalar_summary("T_accuracy", self.accuracy)

        # optimize
        self.optim = self.get_optimizer()

        self.grad_and_var = self.optim.compute_gradients(self.loss)
        with tf.name_scope('clip_norm'):
            new = []
            for _g, v in self.grad_and_var:
                if _g is not None:
                    new.append( (tf.clip_by_norm(_g, self.max_norm), v) )
                else:
                    new.append( (_g,v) )

            self.grad_and_var = new                            

        if not parallel:
            self.train_op = self.optim.apply_gradients(
                self.grad_and_var, name='train_op')
        else:
            self.train_op = None

        # train_sum
        gv_sum = []
        zf = []
        for g, v in self.grad_and_var:
            v_sum = tf.scalar_summary(
                "I_{}-var/mean".format(v.name), tf.reduce_mean(v))
            gv_sum.append(v_sum)
            if g is not None:
                g_sum = tf.scalar_summary(
                    "I_{}-grad/mean".format(v.name), tf.reduce_mean(g))
                zero_frac = tf.scalar_summary(
                    "I_{}-grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                gv_sum.append(g_sum)
                zf.append(zero_frac)

        self.vname = [v.name for g, v in self.grad_and_var]
        self.vars = [v for g, v in self.grad_and_var]
        self.gras = [g for g, v in self.grad_and_var if g is not None]
        self.gname = [ v.name for g, v in self.grad_and_var if g is not None ]

        if self.attention == 'local':
            self.train_sum = tf.merge_summary([loss_sum, acc_sum])
        else:
            self.train_sum = tf.merge_summary([loss_sum, acc_sum])

        # validation sum
        v_loss_sum = tf.scalar_summary("V_loss", tf.reduce_mean(self.loss))
        v_acc_sum = tf.scalar_summary("V_accuracy", self.accuracy)

        embed_sum = tf.histogram_summary("embed", self.emb)
        self.validate_sum = tf.merge_summary(
            [embed_sum, v_loss_sum, v_acc_sum])

    def apply_attention( self, _type, size, d_t, u, local_D=25):

        if _type == 'concat':
            r = self.concat_attention(size, d_t, u)
        elif _type == 'bilinear':
            r = self.bilinear_attention(size, d_t, u)
        elif _type == 'local':
            # r = self.local_attention(d_t, u, attention='concat')

            WT_dm = tf.get_variable('WT_dm', [ size, size])
            WT_um = tf.get_variable('WT_um', [ size, size])
            _u = tf.matmul( u, WT_um )
            _dt = tf.reduce_max( d_t, 1, name='local_dt') # N, 2H
            _dt = tf.matmul( _dt, WT_dm )
            decoder_state = tf.concat( 1, [_u, _dt] )

            content_func = lambda x, y : self.concat_attention(x, u, return_attention=True)
            r, atten_hist = local_attention( decoder_state , d_t, 
                            window_size=local_D, content_function=content_func)
        else:
            raise ValueError(_type)

        return r

    def concat_attention( self, size, d_t, u, return_attention=False):
        W_ym = tf.get_variable('W_ym', [ size, size])
        W_um = tf.get_variable('W_um', [ size, size])
        W_ms = tf.get_variable('W_ms', [ size ])
        m_t = []
        U = tf.matmul(u, W_um)  # N,H

        d_t = tf.unpack(d_t, axis=1)
        for d in d_t:
            m_t.append(tf.matmul(d, W_ym) + U)  # N,H
        m = tf.pack(m_t, 1)  # N,T,H
        m = tf.tanh(m)
        ms = tf.reduce_sum(m * W_ms, 2, keep_dims=True, name='ms')  # N,T,1
        s = tf.nn.softmax(ms, 1)  # N,T,1
        atten = tf.squeeze(s, [-1], name='attention')
        d = tf.pack(d_t, axis=1)  # N,T,2E
        if return_attention:
            return atten
        else:
            r = tf.reduce_sum(s * d, 1, name='r')  # N, 2E
            return r

    def bilinear_attention( self, size, d_t, u, return_attention=False):
        W = tf.get_variable('W_bilinear', [ size, size ])
        atten = []

        d_t = tf.unpack(d_t, axis=1)
        for d in d_t:
            a = tf.matmul(d, W, name='dW')  # N, 2H
            a = tf.reduce_sum(a * u, 1, name='Wq')  # N
            atten.append(a)
        atten = tf.pack(atten, axis=1, )  # N, T
        atten = tf.nn.softmax(atten, name='attention')
        atten = tf.expand_dims(atten, 2)  # N, T, 1
        if return_attention:
            return atten
        else:
            d = tf.pack(d_t, axis=1)
            r = tf.reduce_sum(atten * d, 1, name='r')
            return r

    def _extract_state(self, state, seq_end, need_unpack=True):
        if need_unpack:
            state = tf.unpack(state, axis=0)
        begins = tf.pack(
            [seq_end - 1, tf.zeros([self.batch_size], dtype=tf.int32)], axis=1)  # N, 2
        begins = tf.unpack(begins)

        size = state[0].get_shape()[-1].value

        final_state = []
        for h, b in zip(state, begins):
            f = tf.slice(h, b, [1, size])
            final_state.append(f)

        final = tf.pack(final_state)
        final = tf.squeeze(final, [1])
        return final

    def extract_rnn_state(self, bidirection, state, seq_end):
        if bidirection:
            f = self._extract_state(state[0], seq_end, need_unpack=True)
            bhidden = tf.reverse(state[1], [True, False, True], name='reverse_bw')
            b = self._extract_state( bhidden, seq_end )
            final = tf.concat(1, [f, b])  # N, Hidden*2
        else:
            final = self._extract_state( state, need_unpack=False )

        return final

    def rnn(self, hidden_size, input_tensor, seq_length, dtype=tf.float32, use_bidirection=True, cell_type='LSTM'):

        if cell_type == 'LSTM':
            cell = lambda size: rnn_cell.LSTMCell(size, state_is_tuple=True)
        elif cell_type == 'GRU':
            cell = rnn_cell.GRU
        else:
            raise ValueError(cell_type)

        if use_bidirection:            
            h_t, final_state, = tf.nn.bidirectional_dynamic_rnn(
                cell(hidden_size),
                cell(hidden_size),
                input_tensor,
                sequence_length=seq_length, dtype=dtype)

        else:
            h_t, final_state = tf.nn.dynamic_rnn(
                cell(2 * hidden_size),
                input_tensor,
                sequence_length=seq_length, dtype=dtype)
        return h_t, final_state

    def get_optimizer(self):
        if self.use_optimizer == 'SGD':
            optim = tf.train.GradientDescentOptimizer(
                self.learning_rate, name='optimizer')
        elif self.use_optimizer == 'Adam':
            optim = tf.train.AdamOptimizer(
                self.learning_rate, name='optimizer')
        elif self.use_optimizer == 'RMS':
            optim = tf.train.RMSPropOptimizer(
                self.learning_rate, momentum=self.momentum, decay=self.decay, name='optimizer')
        return optim


    def train(self, sess, vocab_size, epoch=25, data_dir="data", dataset_name="cnn",
              log_dir='log/tmp/', load_path=None, data_size=3000, eval_every=1500, val_rate=0.1, dropout_rate=0.9):

        print(" [*] Building Network...")
        start = time.time()
        self.prepare_model()
        print(" [*] Preparing model finished. Use %4.4f" %
              (time.time() - start))

        # Summary
        writer = tf.train.SummaryWriter(log_dir, sess.graph)
        print(" [*] Writing log to %s" % log_dir)

        # Saver and Load
        self.saver = tf.train.Saver(max_to_keep=15)
        if load_path is not None:
            if os.path.isdir(load_path):
                fname = tf.train.latest_checkpoint(
                    os.path.join(load_path, 'ckpts'))
                assert fname is not None
            else:
                fname = load_path

            print(" [*] Loading %s" % fname)
            self.saver.restore(sess, fname)
            print(" [*] Checkpoint is loaded.")
        else:
            sess.run(tf.initialize_all_variables())
            print(" [*] No checkpoint to load, all variable inited")

        counter = 0
        vcounter = 0
        start_time = time.time()
        ACC = []
        LOSS = []
        train_files, validate_files = fetch_files(
            data_dir, dataset_name, vocab_size)
        if data_size:
            train_files = train_files[:data_size]
        validate_size = int(
            min(max(20.0, float(len(train_files)) * val_rate), len(validate_files)))
        print(" [*] Validate_size %d" % validate_size)

        for epoch_idx in xrange(epoch):
            # load data
            train_iter = data_iter(train_files, self.max_nsteps, self.max_query_length,
                                   batch_size=self.batch_size,
                                   vocab_size=self.vocab_size,
                                   shuffle_data=True)
            tsteps = train_iter.next()

            # train
            running_acc = 0
            running_loss = 0
            for batch_idx, docs, d_end, queries, q_end, y in train_iter:
                _, summary_str, cost, accuracy, gs = sess.run(
                        [self.train_op, self.train_sum, self.loss, self.accuracy,
                            self.gras,
                        ],
                        feed_dict={ self.document: docs,
                                    self.query: queries,
                                    self.d_end: d_end,
                                    self.q_end: q_end,
                                    self.y: y,
                                    self.dropout: dropout_rate,
                                     })

                writer.add_summary(summary_str, counter)
                running_acc += accuracy
                running_loss += np.mean(cost)
                if counter % 10 == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f"
                          % (epoch_idx, batch_idx, tsteps, time.time() - start_time, running_loss / 10.0, running_acc / 10.0))
                    running_loss = 0
                    running_acc = 0
                counter += 1

                if False:
                    for name, g in zip(self.gname, gs):
                        _n = norm(g)
                        if _n > self.max_norm:
                            print 'EXPLODE %s %f' % ( name, _n )

                if (counter + 1) % eval_every == 0:
                    # validate
                    running_acc = 0
                    running_loss = 0

                    idxs = np.random.choice(
                        len(validate_files), size=validate_size)
                    files = [validate_files[idx] for idx in idxs]
                    validate_iter = data_iter(files, self.max_nsteps, self.max_query_length,
                                              batch_size=self.batch_size,
                                              vocab_size=self.vocab_size,
                                              shuffle_data=True)
                    vsteps = validate_iter.next()

                    for batch_idx, docs, d_end, queries, q_end, y in validate_iter:
                        validate_sum_str, cost, accuracy = sess.run([self.validate_sum, self.loss, self.accuracy],
                                                                    feed_dict={self.document: docs,
                                                                               self.query: queries,
                                                                               self.d_end: d_end,
                                                                               self.q_end: q_end,
                                                                               self.y: y,
                                                                               self.dropout: 1.0,
                                                                               })
                        writer.add_summary(validate_sum_str, vcounter)
                        running_acc += accuracy
                        running_loss += np.mean(cost)
                        vcounter += 1

                    ACC.append(running_acc / vsteps)
                    LOSS.append(running_loss / vsteps)
                    vcounter += vsteps
                    print("Epoch: [%2d] Validation time: %4.4f, loss: %.8f, accuracy: %.8f"
                          % (epoch_idx, time.time() - start_time, running_loss / vsteps, running_acc / vsteps))

                    # save
                    self.save(sess, log_dir, global_step=counter)

            print('\n\n')

    def save(self, sess, log_dir, global_step=None):
        assert self.saver is not None
        print(" [*] Saving checkpoints...")
        checkpoint_dir = os.path.join(log_dir, "ckpts")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        fname = os.path.join(checkpoint_dir, 'model')
        self.saver.save(sess, fname, global_step=global_step)