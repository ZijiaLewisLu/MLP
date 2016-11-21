from tensorflow.python.ops.rnn_cell import RNNCell, LSTMCell, GRUCell
import tensorflow as tf

class Match_cell(RNNCell):

    def __init__(self, num_units, share_param, **kwargs):
        super(Match_cell, self).__init__()

        self.cell_type = kwargs.pop('cell_type', 'LSTM')
        if self.cell_type == 'LSTM':
            self.cell = LSTMCell(num_units, **kwargs)
        elif self.cell_type == 'GRU':
            self.cell = GRUCell(num_units, **kwargs)
        else:
            raise ValueError(self.cell_type)

        self.num_units = num_units
        self.share_param = share_param

    def __call__(self, inputs, state, scope=None):
        # print inputs.get_shape()
        # print state.h.get_shape()        
        # import ipdb; ipdb.set_trace()
        Wp, Wr, Bg, WHq, Wt, Ba, H_q = self.share_param
        h = state.h
        G = tf.matmul(inputs, Wp) + tf.matmul(h, Wr) + Bg
        G = tf.tanh(WHq + tf.expand_dims(G, 1))  # N,Q,Hidden_size
        a = tf.nn.softmax(tf.reduce_sum(G * Wt, 2) + Ba)  # N,Q+1

        Hqa = tf.reduce_sum(tf.expand_dims(a, -1) *H_q, 1)  # N,Hidden_size
        z = tf.concat(1, [inputs, Hqa])

        return self.cell(z, state)  # N,Hidden_size

    @property
    def state_size(self):  
        return self.cell.state_size

    @property
    def output_size(self):
        return self.cell.output_size


class Pointer_cell(RNNCell):

    def __init__(self, num_units, share_param, **kwargs):
        super(Pointer_cell, self).__init__()

        self.cell_type = kwargs.pop('cell_type', 'LSTM')
        if self.cell_type == 'LSTM':
            self.cell = LSTMCell(num_units, **kwargs)
        elif self.cell_type == 'GRU':
            self.cell = GRUCell(num_units, **kwargs)
        else:
            raise ValueError(self.cell_type)

        self.num_units = num_units
        self.share_param = share_param
        self.score = []

    def __call__(self, inputs, state, scope=None):
        # print inputs.get_shape()
        Wf, Bf, HV, vt, c, H = self.share_param
        h = state.h
        # print 'h', h.get_shape()
        F  = tf.matmul(h,Wf)+Bf # N, H
        F  = tf.tanh( HV+tf.expand_dims(F,1) ) # N,P,H
        beta = tf.reduce_sum( F*vt, 2 )+c
        beta = tf.nn.softmax( beta ) # N,P
        # print 'beta', beta.get_shape()
        self.score.append(beta)

        inputs = tf.reduce_sum(H*tf.expand_dims(beta, 2), 1)
        # print inputs.get_shape()
        new_h, new_state = self.cell(inputs, state)
        # import ipdb; ipdb.set_trace()
        return new_h, new_state

    @property
    def state_size(self):  
        return self.cell.state_size

    @property
    def output_size(self):
        return self.cell.output_size

