import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from base import BaseModel, apply_attention
import numpy as np

class TestReader(BaseModel):

    def filter_weight(self, half_window_size, embed_size, epsilon=0.2):
        D = half_window_size
        weight = np.zeros( [2*D+1, embed_size, 1, 1] )
        for i in range(D+1):
            weight[D+i,:,:,:] = D-i
            weight[D-i,:,:,:] = D-i

        weight += epsilon
        weight /= weight.sum(axis=0)
        return weight        

    def prepare_model(self, parallel=False):

        self.construct_inputs()

        # Embeding
        self.emb = tf.get_variable("emb", [self.vocab_size, self.size])
        embed_d = tf.nn.embedding_lookup(self.emb, self.document, name='embed_d') # N, sL, E
        embed_q = tf.nn.embedding_lookup(self.emb, self.query, name='embed_q')

        # embed_d = tf.nn.dropout(embed_d, keep_prob=self.dropout)
        # embed_q = tf.nn.dropout(embed_q, keep_prob=self.dropout)

        # representation
        with tf.variable_scope("document_represent"):
            wt = self.filter_weight( self.D, self.size )
            self.filter = tf.constant(wt, dtype=tf.float32, name='cnn_filter')
            # d_t: N, T, Hidden
            inputs = tf.expand_dims( embed_d, -1 ) # N, sL, E, 1
            print inputs.get_shape()
            self.patial_sum = tf.nn.conv2d(inputs, self.filter, 
                                [1, 1, 1, 1], 'SAME', name='conv2d')
            print self.patial_sum.get_shape()
            self.patial_sum = tf.squeeze( self.patial_sum, squeeze_dims=[3] )

        with tf.variable_scope("query_represent"):
            q_t, q_final_state = self.rnn(self.size/2, embed_q, self.q_end, use_bidirection=self.bidirection)

            if self.bidirection:
                q_f = tf.unpack(q_t[0], axis=1)
                q_b = tf.unpack(q_t[-1], axis=1)
                u = tf.concat(1, [q_f[-1], q_b[0]], name='u')  # N, Hidden*2
            else:
                u = tf.unpack(q_t, axis=1)[-1]

            # u = tf.nn.dropout(u, keep_prob=self.dropout)

        print u.get_shape()

        d_t = self.patial_sum
        self.u = u

        # attention
        r = apply_attention(self.attention, self.size, d_t, u, 'concat')

        # predict
        W_rg = tf.get_variable("W_rg", [self.size, self.size])
        W_ug = tf.get_variable("W_ug", [self.size, self.size])
        W_g = tf.get_variable('W_g', [self.size, self.vocab_size])
        mid = tf.matmul(r, W_rg, name='r_x_W') + \
            tf.matmul(u, W_ug, name='u_x_W')
        g = tf.tanh(mid, name='g')
        g = tf.nn.dropout(g, keep_prob=self.dropout)
        g = tf.matmul(g, W_g, name='g_x_W')
        self.score = g

        self.construct_loss_and_summary(self.score)

    
