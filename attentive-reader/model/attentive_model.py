import tensorflow as tf
# from tensorflow.python.ops import rnn_cell
from base import BaseModel


class AttentiveReader(BaseModel):
    """Attentive Reader."""

    def prepare_model(self, parallel=False):

        self.construct_inputs()

        # Embeding
        self.emb = tf.get_variable("emb", [self.vocab_size, self.size])
        embed_d = tf.nn.embedding_lookup(self.emb, self.document, name='embed_d')
        embed_q = tf.nn.embedding_lookup(self.emb, self.query, name='embed_q')

        embed_d = tf.nn.dropout(embed_d, keep_prob=self.dropout)
        embed_q = tf.nn.dropout(embed_q, keep_prob=self.dropout)

        # representation
        with tf.variable_scope("document_represent"):
            # d_t: N, T, Hidden
            d_t, d_final_state = self.rnn( self.size, embed_d, self.d_end, use_bidirection=self.bidirection)
            d_t = tf.concat(2, d_t)

        with tf.variable_scope("query_represent"):
            q_t, q_final_state = self.rnn( self.size, embed_q, self.q_end, use_bidirection=self.bidirection)
            u = self.extract_rnn_state( self.bidirection, q_t, self.q_end )


        d_t = tf.nn.dropout(d_t, keep_prob=self.dropout)
        u = tf.nn.dropout(u, keep_prob=self.dropout)
        self.d_t = d_t
        self.u = u

        # attention
        r = self.apply_attention(self.attention, 2*self.size, d_t, u, 'concat')

        # predict
        W_rg = tf.get_variable("W_rg", [2 * self.size, self.size])
        W_ug = tf.get_variable("W_ug", [2 * self.size, self.size])
        W_g = tf.get_variable('W_g', [self.size, self.vocab_size])
        mid = tf.matmul(r, W_rg, name='r_x_W') + \
            tf.matmul(u, W_ug, name='u_x_W')
        g = tf.tanh(mid, name='g')
        g = tf.matmul(g, W_g, name='g_x_W')
        self.score = g

        # beact_sum = tf.scalar_summary(
        #     'before activitation', tf.reduce_mean(mid))
        # afact_sum = tf.scalar_summary(
        #     'before activitation_after', tf.reduce_mean(g))

        self.construct_loss_and_summary(self.score)

   
