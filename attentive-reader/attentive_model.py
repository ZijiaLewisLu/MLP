import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from utils import load_dataset


class AttentiveReader():
    """Attentive Reader."""

    def __init__(self, vocab_size=50003, batch_size=32,
                 learning_rate=1e-4, momentum=0.9, decay=0.95, 
                 size=256,
                 max_nsteps=1000,
                 max_query_length=20,
                 dropout_rate=0.9,
                 use_optimizer = 'RMS',
                 activation='none'
                 ):
        super(AttentiveReader, self).__init__()

        self.size = size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_nsteps = max_nsteps
        self.max_query_length = max_query_length
        self.vocab_size = vocab_size
        self.dropout = dropout_rate
        self.momentum = momentum
        self.decay = decay
        self.use_optimizer = use_optimizer
        self.activation = activation

        self.saver = None

    def prepare_model(self, parallel=False):

        self.document = tf.placeholder(tf.int32, [self.batch_size, self.max_nsteps], name='document')
        self.query = tf.placeholder(tf.int32, [self.batch_size, self.max_query_length], name='query')
        self.d_end = tf.placeholder(tf.int32, self.batch_size, name='docu-end')
        self.q_end = tf.placeholder(tf.int32, self.batch_size, name='quer-end')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.vocab_size], name='Y')

        # Embeding
        self.emb = tf.get_variable("emb", [self.vocab_size, self.size])
        # shape: batch_size, sentence_length, embedding_size
        embed_d = tf.nn.embedding_lookup(self.emb, self.document, name='embed_d')
        # shape: batch_size, sentence_length, embedding_size
        embed_q = tf.nn.embedding_lookup(self.emb, self.query, name='embed_q')
        embed_sum = tf.histogram_summary("embed", self.emb)
        if self.dropout < 1:
            embed_d = tf.nn.dropout(embed_d, keep_prob=self.dropout)
            embed_q = tf.nn.dropout(embed_q, keep_prob=self.dropout)

        # representation
        with tf.variable_scope("document_represent"):
            # d_t: N, T, Hidden
            d_t, d_final_state, = tf.nn.bidirectional_dynamic_rnn(
                rnn_cell.BasicLSTMCell(
                    self.size, forget_bias=0.0, state_is_tuple=True),
                rnn_cell.BasicLSTMCell(
                    self.size, forget_bias=0.0, state_is_tuple=True),
                embed_d, 
                sequence_length=self.d_end, dtype=tf.float32)
            d_t = tf.concat(2, d_t)
            if self.dropout < 1:
                d_t = tf.nn.dropout(d_t, keep_prob=self.dropout)
            d_t = tf.unpack(d_t, axis=1)

        with tf.variable_scope("query_represent"):
            q_t, q_final_state, = tf.nn.bidirectional_dynamic_rnn(
                rnn_cell.BasicLSTMCell(
                    self.size, forget_bias=0.0, state_is_tuple=True),
                rnn_cell.BasicLSTMCell(
                    self.size, forget_bias=0.0, state_is_tuple=True),
                embed_q, 
                sequence_length=self.q_end, dtype=tf.float32)
            q_f = tf.unpack(q_t[0], axis=1)
            q_b = tf.unpack(q_t[-1], axis=1)
            u = tf.concat(1, [q_f[-1], q_b[0]], name='u')
            if self.dropout < 1:
                u = tf.nn.dropout(u, keep_prob=self.dropout)
       
        # attention
        W_ym = tf.get_variable('W_ym', [2 * self.size, 1])
        W_um = tf.get_variable('W_um', [2 * self.size, 1])
        m_t = []
        for d in d_t:
            m_cur = tf.tanh(tf.matmul(d, W_ym) + tf.matmul(u, W_um))
            m_t.append(m_cur)
        m = tf.concat(1, m_t)  # N,T
        s = tf.expand_dims(tf.nn.softmax(m), -1)  # N,T,1
        d = tf.pack(d_t, axis=1)  # N,T,2E
        r = tf.reduce_sum(s * d, 1, name='r')  # N,2E

        # predict
        W_rg = tf.get_variable("W_rg", [2 * self.size, self.vocab_size])
        W_ug = tf.get_variable("W_ug", [2 * self.size, self.vocab_size])
        mid = tf.matmul(r, W_rg, name='r_x_W') + tf.matmul(u, W_ug, name='u_x_W')
        if self.activation == 'relu':
            g = tf.nn.relu(mid, name='relu_g')
        elif self.activation == 'tanh':
            g = tf.tanh(mid, name='g')
        elif self.activation == 'none':
            g = mid
        else:
            raise ValueError(self.activation)

        beact_sum = tf.scalar_summary('before activitation', tf.reduce_mean(mid))
        afact_sum = tf.scalar_summary('before activitation_after', tf.reduce_mean(g))

        self.loss = tf.nn.softmax_cross_entropy_with_logits(g, self.y, name='loss')
        loss_sum  = tf.scalar_summary("loss", tf.reduce_mean(self.loss))

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(g, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')
        acc_sum   = tf.scalar_summary("accuracy", self.accuracy)

        # optimize
        if self.use_optimizer == 'SGD':
            self.optim = tf.train.GradientDescentOptimizer(self.learning_rate, name='optimizer')
        elif self.use_optimizer == 'Adam':
            self.optim = tf.train.AdamOptimizer(self.learning_rate, name='optimizer')
        elif self.use_optimizer == 'RMS':
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate, momentum=self.momentum, decay=self.decay)

        self.grad_and_var = self.optim.compute_gradients(self.loss)
        if not parallel:
            self.train_op = self.optim.apply_gradients(self.grad_and_var, name='train_op')
        else:
            self.train_op = None

        # train_sum
        gv_sum = []
        for g, v in self.grad_and_var:
            v_sum = tf.scalar_summary( "I_{}-var/mean".format(v.name), tf.reduce_mean(v) )
            gv_sum.append(v_sum)
            if g is not None:
                g_sum = tf.scalar_summary( "I_{}-grad/mean".format(v.name), tf.reduce_mean(g) )
                gv_sum.append(g_sum)
        self.vname = [ v.name for g,v in self.grad_and_var ]
        self.vars  = [ v for g,v in self.grad_and_var ]
        self.gras  = [ g for g,v in self.grad_and_var ]
        self.train_sum = tf.merge_summary([beact_sum, loss_sum, acc_sum, afact_sum] + gv_sum)

        # validation sum
        v_loss_sum  = tf.scalar_summary("V_loss", tf.reduce_mean(self.loss))
        v_acc_sum   = tf.scalar_summary("V_accuracy", self.accuracy)
        self.validate_sum = tf.merge_summary([embed_sum, v_loss_sum, v_acc_sum, gv_sum])


    def train(self, sess, vocab_size, epoch=25, data_dir="data", dataset_name="cnn",
              log_dir='log/tmp/', load_path=None, data_size=3000):

        print(" [*] Building Network...")
        start = time.time()
        self.prepare_model()
        print(" [*] Preparing model finished. Use %4.4f"% (time.time()-start))

        # Summary
        writer = tf.train.SummaryWriter(log_dir, sess.graph)
        print(" [*] Writing log to %s" % log_dir )

        # Saver and Load
        self.saver = tf.train.Saver()
        if load_path is not None:
            if os.path.isdir(load_path):
                fname = tf.train.latest_checkpoint(os.path.join(load_path, 'ckpts'))
                assert fname is not None 
            else:
                fname = load_path

            print(" [*] Loading %s"% fname)
            self.saver.restore(sess, fname)
            print(" [*] Checkpoint is loaded.")
        else:
            sess.run(tf.initialize_all_variables())
            print(" [*] No checkpoint to load, all variable inited")

        counter = 0
        start_time = time.time()
        ACC = []
        LOSS = []

        for epoch_idx in xrange(epoch):
            # load data
            train_iter, tsteps, validate_iter, vsteps = load_dataset(data_dir, dataset_name, \
                                        vocab_size, self.batch_size, self.max_nsteps, self.max_query_length, size=data_size)
            
            # train
            running_acc = 0
            running_loss = 0 
            for batch_idx, docs, d_end, queries, q_end, y in train_iter:
                _, summary_str, cost, accuracy = sess.run([self.train_op, self.train_sum, self.loss, self.accuracy ],
                                                      feed_dict={self.document: docs,
                                                                 self.query: queries,
                                                                 self.d_end: d_end,
                                                                 self.q_end: q_end,
                                                                 self.y: y, 
                                                                 }) 

                writer.add_summary(summary_str, counter)
                running_acc += accuracy
                running_loss += np.mean(cost)
                if counter % 10 == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f"
                        % (epoch_idx, batch_idx, tsteps, time.time() - start_time, running_loss/10.0, running_acc/10.0))
                    running_loss = 0
                    running_acc = 0
                counter += 1

            # validate
            running_acc = 0
            running_loss = 0 
            for batch_idx, docs, d_end, queries, q_end, y in validate_iter:
                validate_sum_str, cost, accuracy = sess.run([self.validate_sum, self.loss, self.accuracy ],
                                          feed_dict={self.document: docs,
                                                     self.query: queries,
                                                     self.d_end: d_end,
                                                     self.q_end: q_end,
                                                     self.y: y, 
                                                     })
                writer.add_summary(validate_sum_str, counter)
                running_acc += accuracy
                running_loss += np.mean(cost)

            ACC.append(running_acc/vsteps)
            LOSS.append(running_loss/vsteps)
            print("Epoch: [%2d] Validation time: %4.4f, loss: %.8f, accuracy: %.8f"
                      %(epoch_idx, time.time()-start_time, running_loss/vsteps, running_acc/vsteps))

            # save
            if (epoch_idx+1) % 3 == 0:
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

