import tensorflow as tf
from tensorflow.python.ops import rnn_cell
# from tensorflow.python.ops.rnn_cell import RNNCell


class AttentionCell(rnn_cell.RNNCell):

    def __init__(self, num_units, query_state, input_size=None, activation='tanh'):
        self.num_units = num_units
        self.activation = activation
        self.query_state = query_state
        assert query_state.get_shape()[-1].value == num_units

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, inputs, state, scope=None):
        Wi = tf.get_variable('W_input',
                             [self.num_units, self.num_units])
        Wh = tf.get_variable('W_hidden',
                             [self.num_units, self.num_units])
        B = tf.get_variable('Bias', [self.num_units])
        i = tf.matmul(inputs, Wi)
        h = tf.matmul(state, Wh)
        h = tf.tanh(h + i + self.query_state + B)
        return h, h


class ML_Attention(object):

    def __init__(self, batch_size, sN, sL, qL,
                 vocab_size, embed_size, hidden_size,
                 learning_rate=5e-3,
                 l2_rate=5e-3,
                 optim='Adam',
                 attention='bilinear',
                 attention_layer=3,
                 glove=False,
                 train_glove=False,
                 max_norm=1.5):
        """
        sN: sentence number 
        sL: sentence length
        qL: query length
        """
        self.passage = tf.placeholder(
            tf.int32, [batch_size, sN, sL], name='passage')
        # self.p_len   = tf.placeholder(tf.int32, [batch_size, sN], name='p_len')
        self.query = tf.placeholder(tf.int32, [batch_size, qL], name='query')
        # self.q_len   = tf.placeholder(tf.int32, [batch_size], name='q_len')
        self.answer = tf.placeholder(tf.int64, [batch_size, sN], name='answer')
        self.dropout = tf.placeholder(tf.float32, name='dropout_rate')

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(
            learning_rate, global_step, 700, 0.95)
        self.lr_sum = tf.scalar_summary('learning_rate', learning_rate)

        self.emb = tf.get_variable(
            "emb", [vocab_size, embed_size], trainable=(not glove or train_glove))
        embed_p = tf.nn.embedding_lookup(
            self.emb, self.passage, name='embed_p')  # N,sN,sL,E
        embed_q = tf.nn.embedding_lookup(
            self.emb, self.query, name='embed_q')  # N,qL,E
        self.embed_sum = tf.histogram_summary("embed", self.emb)

        query_token = tf.unpack(embed_q, axis=1)
        with tf.variable_scope("query_represent"):
            q_rep, final_state_fw, final_state_bw = tf.nn.bidirectional_rnn(
                rnn_cell.LSTMCell(
                    hidden_size, forget_bias=0.0, state_is_tuple=True),
                rnn_cell.LSTMCell(
                    hidden_size, forget_bias=0.0, state_is_tuple=True),
                query_token, dtype=tf.float32)
            _, bfinal = tf.split(1, 2, q_rep[0])
            ffinal, _ = tf.split(1, 2, q_rep[-1])
            q_rep = tf.concat(1, [ffinal, bfinal])

        bow_p = tf.reduce_sum(embed_p, 2)  # N, sN, E
        sentence = tf.unpack(bow_p, axis=1)  # [ N,E ] * sN
        with tf.variable_scope("passage_represent"):
            p_rep, final_state_fw, final_state_bw = tf.nn.bidirectional_rnn(
                rnn_cell.LSTMCell(
                    hidden_size, forget_bias=0.0, state_is_tuple=True),
                rnn_cell.LSTMCell(
                    hidden_size, forget_bias=0.0, state_is_tuple=True),
                sentence,
                dtype=tf.float32,
                # initial_state_fw=final_state_fw,
                # initial_state_bw=final_state_bw,
            )

        with tf.name_scope('REP_dropout'):
            q_rep = tf.nn.dropout(q_rep, self.dropout)
            p_rep = [tf.nn.dropout(p, self.dropout) for p in p_rep]

        if attention == 'bilinear':
            atten = self.bilinear_attention(hidden_size, sN, p_rep, q_rep)
        elif attention == 'concat':
            atten = self.concat_attention(hidden_size, sN, p_rep, q_rep)
        elif attention == 'rnn':
            atten = self.rnn_attention(
                hidden_size, sN, p_rep, q_rep, layer=attention_layer)
        else:
            raise ValueError(attention)

        self.score = atten  # N, sN
        self.alignment = tf.nn.softmax(atten, name='alignment')
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            self.score, self.answer, name='loss')

        if l2_rate > 0:
            with tf.name_scope('l2_reg'):
                for v in tf.trainable_variables():
                    if v.name.endswith('Matrix:0') or v.name.startswith('W'):
                        self.loss += l2_rate * \
                            tf.nn.l2_loss(v, name="%s-l2loss" % v.name[:-2])

        prediction = tf.argmax(self.score, 1)
        answer_id = tf.argmax(self.answer, 1)
        self.correct_prediction = tf.equal(prediction, answer_id)
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float16), name='accuracy')

        if optim == 'Adam':
            self.optim = tf.train.AdamOptimizer(learning_rate)
        elif optim == 'SGD':
            self.optim = tf.train.GradientDescentOptimizer(learning_rate)
        elif optim == 'RMS':
            self.optim = tf.train.RMSPropOptimizer(learning_rate)
        else:
            raise ValueError(optim)

        gvs = self.optim.compute_gradients(self.loss)
        with tf.name_scope('clip_norm'):
            self.gvs = [(tf.clip_by_norm(g, max_norm), v) for g, v in gvs]

        # import ipdb; ipdb.set_trace()

        self.train_op = self.optim.apply_gradients(
            self.gvs, global_step=global_step, name='train_op')
        self.check_op = tf.add_check_numerics_ops()

        # summary ===========================
        accu_sum = tf.scalar_summary('T_accuracy', self.accuracy)
        loss_sum = tf.scalar_summary('T_loss', tf.reduce_mean(self.loss))

        pv_sum = tf.histogram_summary('Var_prediction', prediction)
        av_sum = tf.histogram_summary('Var_answer', answer_id)

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

        self.train_summary = tf.merge_summary(
            [accu_sum, loss_sum, self.embed_sum, gv_sum, gv_hist_sum, self.lr_sum])

        Vaccu_sum = tf.scalar_summary('V_accuracy', self.accuracy)
        Vloss_sum = tf.scalar_summary('V_loss', tf.reduce_mean(self.loss))
        self.validate_summary = tf.merge_summary(
            [Vaccu_sum, Vloss_sum, pv_sum, av_sum])

        # store param =======================
        self.p_rep = p_rep
        self.q_rep = q_rep
        self.embed_p = embed_p
        self.embed_q = embed_q
        self.prediction = prediction
        self.global_step = global_step
        self.gv_sum = gv_sum
        self.origin_gv = gvs
        self.learning_rate = learning_rate

    def bilinear_attention(self, hidden_size, sN, p_rep, q_rep):
        # a[i] = p_rep[i] * W * q_rep
        with tf.variable_scope("bilinear_attention"):
            W = tf.get_variable('W', [2 * hidden_size, 2 * hidden_size])
            atten = []
            for i in range(sN):
                a = tf.matmul(p_rep[i], W, name='pW')  # N, 2H
                a = tf.reduce_sum(a * q_rep, 1, name='Wq')  # N
                atten.append(a)
            atten = tf.pack(atten, axis=1, name='attention')  # N, sN
        return atten

    def concat_attention(self, hidden_size, sN, p_rep, q_rep):
        # a[i] = Ws * tanh( p_rep[i]*Wp + q_rep*Wq )
        with tf.variable_scope("concat_attention"):
            Wp = tf.get_variable('Wp', [2 * hidden_size, 2 * hidden_size])
            Wq = tf.get_variable('Wq', [2 * hidden_size, 2 * hidden_size])
            Ws = tf.get_variable('Ws', [2 * hidden_size])
            atten = []
            Q = tf.matmul(q_rep, Wq, name='q_Wq')
            for i in range(sN):
                a = tf.tanh(tf.matmul(p_rep[i], Wp) + Q)  # N, 2H
                atten.append(a)
            atten = tf.pack(atten, axis=1)  # N, sN, 2H
            atten = tf.reduce_sum(atten * Ws, 2, name='attention')  # N, sN
        return atten

    def rnn_attention(self, hidden_size, sN, p_rep, q_rep, layer=3):
        with tf.variable_scope("rnn_attention") as scope:
            Wq = tf.get_variable('Wq', [2 * hidden_size, 2 * hidden_size])
            Ws = tf.get_variable('Ws', [2 * hidden_size])
            Q = tf.matmul(q_rep, Wq, name='q_Wq')

            atten = []
            for i in range(sN):
                if i > 0:
                    scope.reuse_variables()
                fh, fstate = tf.nn.rnn(                # N, 2H
                    AttentionCell(2 * hidden_size, Q),
                    [p_rep[i]] * layer, dtype=tf.float32)
                atten.append(fh[-1])

            atten = tf.pack(atten, axis=1)  # N, sN, 2H
            atten = tf.reduce_sum(atten * Ws, 2, name='attention')  # N, sN
            return atten
