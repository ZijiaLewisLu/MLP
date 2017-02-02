import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from base import BaseModel
import numpy as np


class LSTMReader(BaseModel):
    """Attentive Reader."""

    def construct_inputs(self):
        self.text = tf.placeholder(
            tf.int32, [self.batch_size, self.max_nsteps + self.max_query_length + 1], name='text')
        self.text_end = tf.placeholder(tf.int32, self.batch_size, name='text-end')
        self.y = tf.placeholder(
            tf.float32, [self.batch_size, self.vocab_size], name='Y')
        self.dropout = tf.placeholder(tf.float32, name='dropout_rate')

    def prepare_model(self, parallel=False):

        self.construct_inputs()

        # Embeding
        self.emb = tf.get_variable("emb", [self.vocab_size, self.size])
        embed = tf.nn.embedding_lookup(self.emb, self.text, name='embed_d')
        embed = tf.nn.dropout( embed, keep_prob=self.dropout)


        self.cell = rnn_cell.BasicLSTMCell(self.size, forget_bias=0.0)
        self.cell = rnn_cell.DropoutWrapper( self.cell, output_keep_prob=self.dropout )
        self.stacked_cell = rnn_cell.MultiRNNCell([self.cell] * 2)

        # self.initial_state = self.stacked_cell.zero_state( self.batch_size, tf.float32)

        hidden, states = tf.nn.bidirectional_dynamic_rnn(
                                            self.stacked_cell, self.stacked_cell,
                                            embed,
                                            sequence_length=self.text_end,
                                            dtype=tf.float32,
                                            )

        final = self.extract_rnn_state( True, hidden, self.text_end )

        final = tf.nn.dropout(final, keep_prob=self.dropout)
        self.final = final

        # predict
        W = tf.get_variable("W", [2 * self.size, self.vocab_size])
        B = tf.get_variable("B", [self.vocab_size])
        g = tf.matmul(final, W, name='g_x_W') + B
        self.score = g

        self.construct_loss_and_summary(self.score)

    def step(self, sess, data, fetch, dropout_rate):
        """sess, data, fetch"""

        # use stop id as delimiter, which is 2

        batch_idx, docs, d_end, queries, q_end, y = data

        text = np.zeros( [self.batch_size, self.max_nsteps + self.max_query_length + 1], dtype=np.int )
        end = np.zeros( self.batch_size, dtype=np.int )
        for b in range(self.batch_size):
            ql = q_end[b]
            dl = d_end[b]
            text[b, :ql] = queries[b, :ql]
            text[b, ql] = 2
            text[b, ql+1:ql+1+dl] = docs[b,:dl]
            end[b] = ql+dl+1

        rslt = sess.run( fetch,
                feed_dict={ self.text: text,
                            self.text_end: end,
                            self.y: y,
                            self.dropout: dropout_rate,
                             }
                           )
        return rslt

   
 