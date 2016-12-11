import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from base import BaseModel

class RRNN_Attention(BaseModel):

    def __init__(self, batch_size, sN, sL, qL,
                 vocab_size, embed_size, hidden_size,
                 learning_rate=5e-3,
                 l2_rate=5e-3,
                 optim_type='Adam',
                 attention_type='bilinear',
                 attention_layer=3,
                 glove=False,
                 train_glove=False,
                 max_norm=1.5):
        """
        sN: sentence number  10
        sL: sentence length  50
        qL: query length     15   
        """
        self.passage = tf.placeholder(
            tf.int32, [batch_size, sN, sL], name='passage')
        self.p_len  = tf.placeholder(tf.int32, [batch_size, sN], name='p_len')
        self.query  = tf.placeholder(tf.int32, [batch_size, qL], name='query')
        self.q_len  = tf.placeholder(tf.int32, [batch_size], name='q_len')
        self.answer = tf.placeholder(tf.int64, [batch_size, sN], name='answer')
        self.dropout = tf.placeholder(tf.float32, name='dropout_rate')

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(
            learning_rate, global_step, 1000, 0.95)
        self.lr_sum = tf.scalar_summary('learning_rate', learning_rate) 

        self.emb = tf.get_variable(
            "emb", [vocab_size, embed_size], trainable=(not glove or train_glove))
        embed_p = tf.nn.embedding_lookup(
            self.emb, self.passage, name='embed_p')  # N,sN,sL,E
        embed_q = tf.nn.embedding_lookup(
            self.emb, self.query, name='embed_q')  # N,qL,E
        self.embed_sum = tf.histogram_summary("embed", self.emb)

        # query_token = tf.unpack(embed_q, axis=1)
        query_token = embed_q
        with tf.variable_scope("query_represent"):
            q_rep, final_state = tf.nn.bidirectional_dynamic_rnn(
                rnn_cell.LSTMCell(
                    hidden_size, forget_bias=0.0, state_is_tuple=True),
                rnn_cell.LSTMCell(
                    hidden_size, forget_bias=0.0, state_is_tuple=True),
                query_token,
                sequence_length=self.q_len,
                dtype=tf.float32)

            ffinal = tf.reduce_max(q_rep[0], [1])
            bfinal = tf.reduce_max(q_rep[-1], [1])
            q_rep = tf.concat(1, [ffinal, bfinal])

        sentence = tf.unpack(embed_p, axis=1)  # [N, sL, E] *sN
        sen_len  = tf.unpack(self.p_len, axis=1) # [N]*sN
        sentence_rep = []
        with tf.variable_scope("sentence_represent") as scope:
            gru_cell = rnn_cell.GRUCell(hidden_size)
            for i, tokens in enumerate(sentence):
                if i > 0:
                    scope.reuse_variables()

                mask = tf.sequence_mask(sen_len[i], sL, dtype=tf.int64)
                _p, _ = tf.nn.dynamic_rnn(  # N, sL, H 
                    gru_cell, tokens, 
                    sequence_length=mask,
                    dtype=tf.float32)

                sentence_rep.append( tf.reduce_max(_p, 1) )  # [N, H] * sL
            sentence = tf.pack(sentence_rep, 1)


        sN_mask = tf.to_float(self.p_len > 0, name='sN_mask')  # N, sN
        sN_count = tf.reduce_sum(sN_mask, 1)
        sN_count = tf.to_int64(sN_count, name='sN_count')
        self.sN_mask = sN_mask
        self.sN_count = sN_count

        with tf.variable_scope("passage_represent"):
            p_rep, final_state = tf.nn.bidirectional_dynamic_rnn(
                rnn_cell.LSTMCell(
                    hidden_size, forget_bias=0.0, state_is_tuple=True),
                rnn_cell.LSTMCell(
                    hidden_size, forget_bias=0.0, state_is_tuple=True),
                sentence,
                sequence_length=sN_count,
                dtype=tf.float32,
            )
            p_rep = tf.concat(2, p_rep)


        with tf.name_scope('REP_dropout'):
            q_rep = tf.nn.dropout(q_rep, self.dropout)
            p_rep = tf.nn.dropout(p_rep, self.dropout)


        p_rep = tf.unpack(p_rep, axis=1)
        atten = self.apply_attention(attention_type, hidden_size, sN, p_rep, q_rep, layer=attention_layer)

        atten = atten - tf.reduce_min(atten, [1], keep_dims=True)
        atten = tf.mul(atten, sN_mask, name='unnormalized_attention')

        self.score = atten  # N, sN
        self.alignment = tf.nn.softmax(atten)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            self.score, self.answer, name='loss')

        self.prediction = tf.argmax(self.score, 1)
        self.answer_id = tf.argmax(self.answer, 1)
        self.correct_prediction = tf.equal(self.prediction, self.answer_id)
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float16))



        self.optim = self.get_optimizer(optim_type, learning_rate )
        gvs = self.optim.compute_gradients(self.loss)
        with tf.name_scope('clip_norm'):
            self.gvs = [ ( tf.clip_by_norm(g, max_norm), v ) for g,v in gvs ]

        self.train_op = self.optim.apply_gradients(
            self.gvs, global_step=global_step)
        self.check_op = tf.add_check_numerics_ops()

        # summary ==========================
        tsum, vsum = self.create_summary()
        self.train_summary = tf.merge_summary(tsum)
        self.validate_summary = tf.merge_summary(vsum)

        # store param =======================
        self.p_rep = p_rep
        self.q_rep = q_rep
        self.embed_p = embed_p
        self.embed_q = embed_q
        self.global_step = global_step
        self.origin_gv = gvs
        self.learning_rate = learning_rate
