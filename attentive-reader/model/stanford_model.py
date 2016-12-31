import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from base import apply_attention, BaseModel


class StanfordReader(BaseModel):

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
            d_t, d_final_state = self.rnn(embed_d, self.d_end, use_bidirection=self.bidirection)
            d_t = tf.concat(2, d_t)
            d_t = tf.nn.dropout(d_t, keep_prob=self.dropout)

        with tf.variable_scope("query_represent"):
            q_t, q_final_state = self.rnn(embed_q, self.q_end, use_bidirection=self.bidirection)

            if self.bidirection:
                q_f = tf.unpack(q_t[0], axis=1)
                q_b = tf.unpack(q_t[-1], axis=1)
                u = tf.concat(1, [q_f[-1], q_b[0]], name='u')  # N, Hidden*2
            else:
                u = tf.unpack(q_t, axis=1)[-1]

            u = tf.nn.dropout(u, keep_prob=self.dropout)


        self.d_t = d_t
        self.u = u

        # attention
        r = apply_attention(self.attention, 2*self.size, d_t, u, 'concat')

        # predict
        W_pred = tf.get_variable(name="W_pred", shape=[self.size*2, self.vocab_size ])        
        B_pred = tf.get_variable(name="B_pred", shape=[self.vocab_size])
        g = tf.matmul(r, W_pred, name='r_x_Wpred') 
        self.score = tf.add( g, B_pred, name='score' )

        self.construct_loss_and_summary(self.score)
