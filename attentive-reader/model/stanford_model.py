import tensorflow as tf
# from tensorflow.python.ops import rnn_cell
from base import BaseModel


class StanfordReader(BaseModel):

    def prepare_model(self, parallel=False):

        self.attention = 'bilinear'

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
            d_t, d_final_state = self.rnn(self.size, embed_d, self.d_end, use_bidirection=self.bidirection)
            d_t = tf.concat(2, d_t)
           
        with tf.variable_scope("query_represent"):
            q_t, q_final_state = self.rnn(self.size, embed_q, self.q_end, use_bidirection=self.bidirection)
            u = self.extract_rnn_state( self.bidirection, q_t, self.q_end )

        # d_t = tf.nn.dropout(d_t, keep_prob=self.dropout)
        # u = tf.nn.dropout(u, keep_prob=self.dropout)
        self.d_t = d_t
        self.u = u

        # attention
        r = self.apply_attention(self.attention, 2*self.size, d_t, u, 'bilinear')

        # predict
        W_pred = tf.get_variable(name="W_pred", shape=[self.size*2, self.vocab_size ])        
        B_pred = tf.get_variable(name="B_pred", shape=[self.vocab_size])
        g = tf.matmul(r, W_pred, name='r_x_Wpred') 
        self.score = tf.add( g, B_pred, name='score' )

        self.construct_loss_and_summary(self.score)



class StanfordReader2(BaseModel):

    def prepare_model(self, parallel=False):

        self.attention = 'bilinear'

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

        # d_t = tf.nn.dropout(d_t, keep_prob=self.dropout)
        # u = tf.nn.dropout(u, keep_prob=self.dropout)
        self.d_t = d_t
        self.u = u

        # attention 
        # N, T, 1
        atten = self.bilinear_attention(2*self.size, d_t, u, return_attention=True)
        r = tf.reduce_sum(atten * embed_d, 1, name='r')  # N, E

        # predict
        W_pred = tf.get_variable(name="W_pred", shape=[self.size, self.vocab_size ])        
        B_pred = tf.get_variable(name="B_pred", shape=[self.vocab_size])
        g = tf.matmul(r, W_pred, name='r_x_Wpred') 
        self.score = tf.add( g, B_pred, name='score' )

        self.construct_loss_and_summary(self.score)