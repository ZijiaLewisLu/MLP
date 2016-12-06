import tensorflow as tf
from tensorflow.python.ops import rnn_cell


class ML_Attention(object):

    def __init__(self, batch_size, sN, sL, qL,
                 vocab_size, embed_size, hidden_size,
                 learning_rate=5e-3,
                 dropout_rate=1,
                 l2_rate=5e-3,
                 optim='Adam',
                 attention='bilinear',
                 glove=False):
        """
        sN: sentence number  10
        sL: sentence length  50
        qL: query length     15   
        """
        self.passage = tf.placeholder(
            tf.int32, [batch_size, sN, sL], name='passage')
        # self.p_len   = tf.placeholder(tf.int32, [batch_size, sN], name='p_len')
        self.query = tf.placeholder(tf.int32, [batch_size, qL], name='query')
        # self.q_len   = tf.placeholder(tf.int32, [batch_size], name='q_len')
        self.answer = tf.placeholder(tf.int64, [batch_size, sN], name='answer')

        self.emb = tf.get_variable(
            "emb", [vocab_size, embed_size], trainable=(not glove))
        embed_p = tf.nn.embedding_lookup(
            self.emb, self.passage, name='embed_p')  # N,sN,sL,E
        embed_q = tf.nn.embedding_lookup(
            self.emb, self.query, name='embed_q')  # N,qL,E
        self.embed_sum = tf.histogram_summary("embed", self.emb)

        sentence = tf.unpack(embed_p, axis=1)  # [N, sL, E] *sN
        sentence_rep = []
        with tf.variable_scope("sentence_represent") as scope:
            gru_cell = rnn_cell.GRUCell(hidden_size)
            for i, tokens in enumerate(sentence):
                if i > 0:
                    scope.reuse_variables()
                tokens = tf.unpack(tokens, axis=1)  # [N, E] * sL
                _p, _ = tf.nn.rnn(
                    gru_cell, tokens, dtype=tf.float32)
                sentence_rep.append(_p[-1])  # [N, H] * sL

        query_token = tf.unpack(embed_q, axis=1)
        with tf.variable_scope("query_represent"):
            q_rep, final_state_fw, final_state_bw = tf.nn.bidirectional_rnn(
                rnn_cell.GRUCell( hidden_size ),
                rnn_cell.GRUCell( hidden_size ),
                query_token, dtype=tf.float32)

            _, bfinal = tf.split(1, 2, q_rep[0])
            ffinal, _ = tf.split(1, 2, q_rep[-1])
            q_rep = tf.concat(1, [ffinal, bfinal])

        # sentence = tf.unpack(bow_p, axis=1)  # [ N,E ] * sN
        # print type(final_state_fw)
        # print final_state_fw
        with tf.variable_scope("passage_represent"):
            p_rep, final_state_fw, final_state_bw = tf.nn.bidirectional_rnn(
                rnn_cell.GRUCell( hidden_size ),
                rnn_cell.GRUCell( hidden_size ),
                sentence_rep,
                dtype=tf.float32,
                initial_state_fw=final_state_fw,
                initial_state_bw=final_state_bw,
            )

        if dropout_rate < 1:
            q_rep = tf.nn.dropout(q_rep, dropout_rate)
            p_rep = [tf.nn.dropout(p, dropout_rate) for p in p_rep]

        if attention == 'bilinear':
            atten = self.bilinear_attention(hidden_size, sN, p_rep, q_rep)
        elif attention == 'concat':
            atten = self.concat_attention(hidden_size, sN, p_rep, q_rep)
        else:
            raise ValueError(attention)

        self.score = atten  # N, sN
        self.alignment = tf.nn.softmax(atten)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            self.score, self.answer, name='loss')
        if l2_rate > 0:
            for v in tf.trainable_variables():
                if v.name.endswith('Matrix:0') or v.name.startswith('W'):
                    self.loss += l2_rate * \
                        tf.nn.l2_loss(v, name="%s-l2loss" % v.name[:-2])

        prediction = tf.argmax(self.score, 1)
        answer_id = tf.argmax(self.answer, 1)
        self.correct_prediction = tf.equal(prediction, answer_id)
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float16))

        if optim == 'Adam':
            self.optim = tf.train.AdamOptimizer(learning_rate)
        elif optim == 'SGD':
            self.optim = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError(optim)
        self.gvs = self.optim.compute_gradients(self.loss)
        self.train_op = self.optim.apply_gradients(self.gvs)
        self.check_op = tf.add_check_numerics_ops()

        # summary ==========================
        pv_sum = tf.histogram_summary('Var_prediction', prediction)
        av_sum = tf.histogram_summary('Var_answer', answer_id)
        self.hist_sum = [self.embed_sum, pv_sum, av_sum]

        accu_sum = tf.scalar_summary('T_accuracy', self.accuracy)
        loss_sum = tf.scalar_summary('T_loss', tf.reduce_mean(self.loss))
        self.train_summary = tf.merge_summary([accu_sum, loss_sum])

        Vaccu_sum = tf.scalar_summary('V_accuracy', self.accuracy)
        Vloss_sum = tf.scalar_summary('V_loss', tf.reduce_mean(self.loss))
        self.validate_summary = tf.merge_summary(
            [Vaccu_sum, Vloss_sum] + self.hist_sum)

        # store param =======================
        self.p_rep = p_rep
        self.q_rep = q_rep
        self.embed_p = embed_p
        self.embed_q = embed_q
        self.prediction = prediction

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
