import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from base import BaseModel


class Attention(BaseModel):

    def __init__(self, batch_size, sN, sL, qL,
                 vocab_size, embed_size, hidden_size,
                 learning_rate=5e-3,
                 optim_type='Adam',
                 attention_type='bilinear',
                 attention_layer=3,
                 glove=False,
                 train_glove=False,
                 max_norm=6):
        """
        sN: sentence number 
        sL: sentence length
        qL: query length

        Placeholders
        # passage [batch_size, sN, sL]
        # p_len   [batch_size, sN]
        # p_idf   [batch_size, sN, sL]
        # query   [batch_size, qL]
        # q_len   [batch_size]
        # q_idf   [batch_size, qL]
        # answer  [batch_size, sN]
        # dropout scalar
        """

        self.create_placeholder(batch_size, sN, sL, qL)

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

        with tf.name_scope('BoW'):
            wt_p = tf.expand_dims( self.p_wt, -1 )
            bow_p = tf.reduce_sum( embed_p*wt_p, 2, name='bow_p' ) # N, sN, E
            epsilon = 1e-5
            denominator = tf.to_float(tf.expand_dims( self.p_len, -1 )) + epsilon
            # bow_p = tf.div( bow_p, denominator, name= 'true_bow_p' ) # N, sN, 1

            wt_q = tf.expand_dims( self.q_wt, -1 ) 
            bow_q = tf.reduce_sum( embed_q*wt_q, 1, name='bow_q') # N, E
            denominator = tf.to_float(tf.expand_dims( self.q_len, -1 )) + epsilon
            # bow_q = tf.div( bow_q, denominator, name='true_bow_q' ) # N, 1

            p_rep = bow_p
            q_rep = bow_q

        sN_mask = tf.to_float(self.p_len > 0, name='sN_mask')  # N, sN
        sN_count = tf.reduce_sum(sN_mask, 1)
        self.sN_mask = sN_mask
        self.sN_count = sN_count
        sN_count = tf.to_int64(sN_count, name='sN_count')
        # self.sn_c_print = tf.Print(sN_count, [sN_count, sN_mask], message='sn count, sn mask', first_n=50)

        with tf.name_scope('REP_dropout'):
            q_rep = tf.nn.dropout(q_rep, self.dropout)
            p_rep = tf.nn.dropout(p_rep, self.dropout)

        p_rep = tf.unpack(p_rep, axis=1)
        atten = self.apply_attention(
            attention_type, embed_size/2, sN, p_rep, q_rep, layer=attention_layer)

        atten = atten - tf.reduce_min(atten, [1], keep_dims=True)
        atten = tf.mul(atten, sN_mask, name='unnormalized_attention')

        self.score = atten  # N, sN
        self.alignment = tf.nn.softmax(atten, name='alignment')

        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            self.score, self.answer, name='loss')

        self.prediction = tf.argmax(self.score, 1)
        self.answer_id = tf.argmax(self.answer, 1)
        self.correct_prediction = tf.equal(
            self.prediction, self.answer_id)  # N
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32), name='accuracy')

        self.optim = self.get_optimizer(optim_type, learning_rate)
        gvs = self.optim.compute_gradients(self.loss)
        with tf.name_scope('clip_norm'):
            self.gvs = [(tf.clip_by_norm(g, max_norm), v) for g, v in gvs]

        self.train_op = self.optim.apply_gradients(
            self.gvs, global_step=global_step, name='train_op')
        self.check_op = tf.add_check_numerics_ops()

        tsum, vsum = self.create_summary(add_gv_sum=True)
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
