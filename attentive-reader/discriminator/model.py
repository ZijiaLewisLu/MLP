# /usr/bin/python
import tensorflow as tf


class Base(object):

    def construct_loss_and_accuracy(self, score, label):
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            score, label, name='loss')

        self.prediction = tf.argmax(score, 1, name='prediction')
        self.right_label = tf.argmax(label, 1, name='right_label')
        self.correct = tf.equal(
            self.prediction, self.right_label, name='correct_prediction')
        self.accuracy = tf.reduce_mean(
            tf.to_float(self.correct), name='accuracy')

        # optim = tf.train.GradientDescentOptimizer(learning_rate)
        self.optim = tf.train.AdamOptimizer(self.learning_rate)

        self.gvs = self.optim.compute_gradients(self.loss)
        with tf.name_scope('clip_norm'):
            self.gvs = [(tf.clip_by_norm(g, self.clip_norm), v)
                        for g, v in self.gvs]

        self.train_op = self.optim.apply_gradients(
            self.gvs, global_step=self.global_step)

    def construct_summary(self, add_gv_sum=True):
        accu_sum = tf.scalar_summary('T_accuracy', self.accuracy)
        loss_sum = tf.scalar_summary('T_loss', tf.reduce_mean(self.loss))

        final_sparsity_sum = tf.scalar_summary(
            'T_final_sparsity', self.final_sparsity)

        train_summary = [accu_sum, loss_sum, final_sparsity_sum]

        gv_sum = []
        gv_hist_sum = []

        for g, v in self.gvs:
            v_sum = tf.scalar_summary(
                "I_{}-var/mean".format(v.name), tf.reduce_mean(v))
            v_his = tf.histogram_summary("I_{}-var".format(v.name), v)

            if g is not None:
                g_sum = tf.scalar_summary(
                    "I_{}-grad/mean".format(v.name), tf.reduce_mean(g))
                zero_frac = tf.scalar_summary(
                    "I_{}-grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                g_his = tf.histogram_summary("I_{}-grad".format(v.name), g)

            gv_sum += [v_sum, g_sum, zero_frac]
            gv_hist_sum += [v_his, g_his]

        if add_gv_sum:
            train_summary += [gv_sum, gv_hist_sum]

        Vaccu_sum = tf.scalar_summary('V_accuracy', self.accuracy)
        Vloss_sum = tf.scalar_summary('V_loss', tf.reduce_mean(self.loss))
        validate_summary = [Vaccu_sum, Vloss_sum]

        self.gv_sum = gv_sum
        self.gv_hist_sum = gv_hist_sum

        self.train_summary = tf.merge_summary(train_summary)
        self.validate_summary = tf.merge_summary(validate_summary)

    def step(self, sess, data, fetch):
        idx, d, dl, l = data
        # prin
        return sess.run(fetch,
                        feed_dict={
                            self.data: d,
                            self.d_len: dl,
                            self.label: l,
                        })


class Discriminator(Base):

    def __init__(self, batch_size, hidden_size,
                 learning_rate=1e-4,
                 num_layer=2,
                 sequence_length=1000,
                 clip_norm=6,
                 reuse=True):

        self.data = tf.placeholder(
            tf.float32, [batch_size, sequence_length, 2], 'data')
        self.d_len = tf.placeholder(tf.int32, [batch_size], 'd_len')
        self.label = tf.placeholder(tf.int32, [batch_size, 3], 'label')
        self.global_step = tf.Variable(1, name='global_step', trainable=False)

        self.clip_norm = clip_norm
        self.learning_rate = learning_rate

        if reuse:
            self.cells = [tf.nn.rnn_cell.LSTMCell(
                hidden_size, state_is_tuple=True)] * num_layer
        else:
            self.cells = [tf.nn.rnn_cell.LSTMCell(
                hidden_size, state_is_tuple=True) for _ in range(num_layer)]

        cell = tf.nn.rnn_cell.MultiRNNCell(self.cells, state_is_tuple=True)

        self.hidden, last_state = tf.nn.dynamic_rnn(
            cell,
            self.data,
            sequence_length=self.d_len,
            dtype=tf.float32,
        )

        with tf.name_scope('final_state'):
            unpack = tf.unpack(self.hidden, axis=0)
            begins = tf.pack(
                [self.d_len - 1, tf.zeros([batch_size], dtype=tf.int32)], axis=1)  # N, 2
            begins = tf.unpack(begins)
            final_state = []
            for h, b in zip(unpack, begins):
                f = tf.slice(h, b, [1, hidden_size])
                print f.get_shape()
                final_state.append(f)

            final = tf.pack(final_state)

            self.final = tf.squeeze(final_state, [1])

        self.final_sparsity = tf.nn.zero_fraction(
            final, name='final_hidden_sparsity')
        self.final_relu = tf.nn.relu(self.final, name='final_relu')

        W = tf.get_variable('W', [hidden_size, 3], dtype=tf.float32)
        self.score = tf.matmul(self.final_relu, W, name='score')

        self.construct_loss_and_accuracy(self.score, self.label)
        self.construct_summary(add_gv_sum=True)


class One_Hot(Base):
    def __init__(self, batch_size, hidden_size,
                 learning_rate=1e-4,
                 num_layer=2,
                 sequence_length=1000,
                 clip_norm=6,
                 vocab_size=50003,
                 reuse=True):

        self.data = tf.placeholder(
            tf.float32, [batch_size, sequence_length, 2], 'data')
        self.d_len = tf.placeholder(tf.int32, [batch_size], 'd_len')
        self.label = tf.placeholder(tf.int32, [batch_size, 3], 'label')
        self.global_step = tf.Variable(1, name='global_step', trainable=False)

        wid, ascore = tf.unpack(self.data, axis=2)  # N, sL
        self.one_hot = tf.one_hot(wid, vocab_size, axis=-1, name='one_hot')
        ascore = tf.expand_dims(ascore, axis=-1)
        self.features = self.one_hot * ascore

        if reuse:
            self.cells = [tf.nn.rnn_cell.LSTMCell(
                hidden_size, state_is_tuple=True)] * num_layer
        else:
            self.cells = [tf.nn.rnn_cell.LSTMCell(
                hidden_size, state_is_tuple=True) for _ in range(num_layer)]

        cell = tf.nn.rnn_cell.MultiRNNCell(self.cells, state_is_tuple=True)

        self.hidden, last_state = tf.nn.dynamic_rnn(
            cell,
            self.features,
            sequence_length=self.d_len,
            dtype=tf.float32,
        )

        with tf.name_scope('final_state'):
            unpack = tf.unpack(self.hidden, axis=0)
            begins = tf.pack(
                [self.d_len - 1, tf.zeros([batch_size], dtype=tf.int32)], axis=1)  # N, 2
            begins = tf.unpack(begins)
            final_state = []
            for h, b in zip(unpack, begins):
                f = tf.slice(h, b, [1, hidden_size])
                final_state.append(f)

            final = tf.pack(final_state)
            self.final = tf.squeeze(final_state, [1])

        self.final_sparsity = tf.nn.zero_fraction(
            final, name='final_hidden_sparsity')
        self.final_relu = tf.nn.relu(self.final, name='final_relu')

        W = tf.get_variable('W', [hidden_size, 3], dtype=tf.float32)
        self.score = tf.matmul(self.final_relu, W, name='score')

        self.construct_loss_and_accuracy(self.score, self.label)
        self.construct_summary(add_gv_sum=True)
