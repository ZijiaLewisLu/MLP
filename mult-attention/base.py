import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import numpy as np

def orthogonal_initializer(scale=1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    print 'Warning -- You have opted to use the orthogonal_initializer function'
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if len(shape) == 1:
            a = np.random.normal(0.0, 1.0, shape[0])
            return tf.constant(scale * a, dtype=tf.float32)

        flat_shape = (shape[0], np.prod(shape[1:], dtype=np.int32))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        # print('you have initialized one orthogonal matrix.')
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer

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


class BaseModel(object):

    def __init__(self, *args, **kwargs):
        pass

    def apply_attention(self, _type, hidden_size, sN, p_rep, q_rep, layer=3):
        print '  Using attention %s' % _type
        if _type == 'bilinear':
            atten = self.bilinear_attention(hidden_size, sN, p_rep, q_rep)
        elif _type == 'concat':
            atten = self.concat_attention(hidden_size, sN, p_rep, q_rep)
        elif _type == 'rnn':
            atten = self.rnn_attention(
                hidden_size, sN, p_rep, q_rep, layer=layer)
        elif _type == 'mlp':
            atten = self.mlp_attention(
                hidden_size, sN, p_rep, q_rep, layer=layer)
        else:
            raise ValueError(_type)

        return atten

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
            self.before_tanh = []
            for i in range(sN):
                pWQ = tf.matmul(p_rep[i], Wp) + Q
                self.before_tanh.append(pWQ)
                a = tf.tanh(pWQ)  # N, 2H
                atten.append(a)
            atten = tf.pack(atten, axis=1)  # N, sN, 2H
            atten = tf.reduce_sum(atten * Ws, 2, name='attention')  # N, sN
        return atten

    def mlp_attention(self, hidden_size, sN, p_rep, q_rep, layer=3):
        with tf.variable_scope('mlp_attention') as scope:
            Wq = tf.get_variable('Wq', [2 * hidden_size, hidden_size])
            Ws = tf.get_variable('Ws', [hidden_size])
            Q = tf.matmul(q_rep, Wq, name='q_Wq')
            atten = []
            for i in range(sN):
                p = p_rep[i]
                if i > 0:
                    scope.reuse_variables()
                for l in range(layer):
                    in_shape = 2*hidden_size if l == 0 else hidden_size
                    Wp_i = tf.get_variable(
                        'Wq%d' % l, [in_shape, hidden_size])
                    B_i = tf.get_variable('%d_B' % l, [hidden_size])

                    pWQ = tf.matmul(p, Wp_i) + Q + B_i
                    p = tf.tanh(pWQ)
                atten.append(p)
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

    def get_optimizer(self, _type, learning_rate):
        if _type == 'Adam':
            optim = tf.train.AdamOptimizer(learning_rate)
        elif _type == 'SGD':
            optim = tf.train.GradientDescentOptimizer(learning_rate)
        elif _type == 'RMS':
            optim = tf.train.RMSPropOptimizer(learning_rate)
        else:
            raise ValueError(_type)
        return optim

    def create_summary(self):
        self.align_his = tf.histogram_summary('alignment', self.alignment)

        accu_sum = tf.scalar_summary('T_accuracy', self.accuracy)
        loss_sum = tf.scalar_summary('T_loss', tf.reduce_mean(self.loss))

        pv_sum = tf.histogram_summary('Var_prediction', self.prediction)
        av_sum = tf.histogram_summary('Var_answer', self.answer_id)

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

        train_summary = [accu_sum, loss_sum, self.embed_sum, gv_sum, gv_hist_sum, self.lr_sum, self.align_his]

        Vaccu_sum = tf.scalar_summary('V_accuracy', self.accuracy)
        Vloss_sum = tf.scalar_summary('V_loss', tf.reduce_mean(self.loss))
        validate_summary = [Vaccu_sum, Vloss_sum, pv_sum, av_sum]

        self.gv_sum = gv_sum
        self.gv_hist_sum = gv_hist_sum

        return train_summary, validate_summary

