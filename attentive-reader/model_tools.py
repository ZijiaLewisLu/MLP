import tensorflow as tf


def apply_attention(self, _type, size, d_t, u, auxi_arg):

    if _type == 'concat':
        r = concat_attention(size, d_t, u)
    elif _type == 'bilinear':
        r = bilinear_attention(size, d_t, u)
    elif _type == 'local':
        # r = self.local_attention(d_t, u, attention='concat')
        from attention import local_attention

        WT_dm = tf.get_variable('WT_dm', [ size, size])
        WT_um = tf.get_variable('WT_um', [ size, size])
        _u = tf.matmul( u, WT_um )
        # _u = tf.matmul( u, W_ym )
        _dt = tf.reduce_max( d_t, 1, name='local_dt') # N, 2H
        _dt = tf.matmul( _dt, WT_dm )
        decoder_state = tf.concat( 1, [_u, _dt] )

        content_func = lambda x, y : concat_attention(x, u, return_attention=auxi_arg)
        r, atten_hist = local_attention( decoder_state , d_t, 
                        window_size=self.D, content_function=content_func)
    else:
        raise ValueError(_type)

    return r

def concat_attention( size, d_t, u, return_attention=False):
    W_ym = tf.get_variable('W_ym', [ size, size])
    W_um = tf.get_variable('W_um', [ size, size])
    W_ms = tf.get_variable('W_ms', [ size ])
    m_t = []
    U = tf.matmul(u, W_um)  # N,H

    d_t = tf.unpack(d_t, axis=1)
    for d in d_t:
        m_t.append(tf.matmul(d, W_ym) + U)  # N,H
    m = tf.pack(m_t, 1)  # N,T,H
    m = tf.tanh(m)
    ms = tf.reduce_sum(m * W_ms, 2, keep_dims=True, name='ms')  # N,T,1
    s = tf.nn.softmax(ms, 1)  # N,T,1
    atten = tf.squeeze(s, [-1], name='attention')
    d = tf.pack(d_t, axis=1)  # N,T,2E
    if return_attention:
        return atten
    else:
        r = tf.reduce_sum(s * d, 1, name='r')  # N, 2E
        return r

def bilinear_attention( size, d_t, u, return_attention=False):
    W = tf.get_variable('W_bilinear', [ size, size ])
    atten = []

    d_t = tf.unpack(d_t, axis=1)
    for d in d_t:
        a = tf.matmul(d, W, name='dW')  # N, 2H
        a = tf.reduce_sum(a * u, 1, name='Wq')  # N
        atten.append(a)
    atten = tf.pack(atten, axis=1, )  # N, T
    atten = tf.nn.softmax(atten, name='attention')
    atten = tf.expand_dims(atten, 2)  # N, T, 1
    d = tf.pack(d_t, axis=1)
    if return_attention:
        return atten
    else:
        r = tf.reduce_sum(atten * d, 1, name='r')
        return r

# def cheap_attention(self, d_t, u, return_attention=False):


# def local_attention( size, d_t, u, attention='bilinear'):

#     Wp = tf.get_variable('Wp', [ size, size])
#     V = tf.get_variable('V', [ size ])
#     D = self.D

#     tanh = tf.tanh(tf.matmul(u, Wp))
#     p = tf.reduce_sum(tanh * V, 1)  # N
#     p = self.max_nsteps * tf.sigmoid(p)
#     p = tf.to_int32(tf.floor(p))
#     self.p = p

#     pt = tf.minimum(p, D)
#     pt = tf.maximum(pt, self.max_nsteps - D - 1)
#     begin_idx = pt - D  # N
#     zero = tf.constant(0, shape=[self.batch_size], dtype=tf.int32)
#     begin = tf.pack([begin_idx, zero], 1)  # N, 2 of [p-D, 0]
#     size = tf.constant([2 * D, -1], dtype=tf.int32)

#     with tf.name_scope('attention_extract'):
#         batches = []  # N * [ 2D*2H ]
#         begin = tf.unpack(begin)
#         d_t = tf.unpack(d_t)
#         for b, d in zip(begin, d_t):
#             block = tf.slice(d, b, size)
#             batches.append(block)

#         batches = tf.pack(batches)

#     if attention == 'bilinear':
#         alignment = self.bilinear_attention(
#             batches, u, return_attention=True)  # N, 2D, 2H
#     elif attention == 'concat':
#         alignment = self.concat_attention(
#             batches, u, return_attention=True)
#     else:
#         raise ValueError(attention)

#     # here we calculate the 'truncated normal distribution'
#     idx = [begin_idx + i for i in range(2 * D)]  # [N]*2D
#     idx = tf.pack(idx, 1)  # N, 2D

#     denominator = (D / 2.0) ** 2.0
#     numerator = -tf.pow(tf.to_float((idx - tf.expand_dims(p, 1))), 2.0)
#     div = tf.truediv(numerator, denominator)
#     e = tf.exp(div)  # result of the truncated normal distribution

#     self.attention = tf.mul(alignment, e, name='local_attention')  # N, 2D
#     r = tf.reduce_sum(
#         batches * tf.expand_dims(self.attention, -1), 1, name='r')
#     return r