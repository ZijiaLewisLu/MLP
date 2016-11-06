import tensorflow as tf

class BASEModel():
    def __init__(self):
        raise NotImplementedError()
    
    def build_model(self):
        raise NotImplementedError()

    def encode_step(self):
        raise NotImplementedError()

    def inference_step(self):
        raise NotImplementedError()

class MatchLSTM(BASEModel):
    
    def __init__(self, vocab_size, q_length, p_length, batch_size, embedding_size=128, hidden_size=128):
        """
        vocab_size, question_length, passage_length, batch_size, embedding_size=128, hidden_size=128
        """
        self._embedding_size = embedding_size
        self._hidden_size = hidden_size
        self._q_length = q_length
        self._voca_size = vocab_size
        self._batch_size = batch_size
        self._p_length = p_length

    def match_unit(self, direction, param):
        Wp, Wr, Bg, WHq, H_p, Wt, Ba, hq_stack = param
        H = []
        with tf.variable_scope(direction) as scope:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self._hidden_size, state_is_tuple=True)
            state = cell.zero_state(self._batch_size, hq_stack.dtype)
            h = state.h
            idx = range(len(H_p))
            if direction == 'backward': idx=list(reversed(idx))

            for i in idx:
                if i != idx[0] :
                    scope.reuse_variables()
                # attention
                G = tf.matmul(H_p[i],Wp)+tf.matmul(h, Wr) + Bg # N, Hidden_size
                G = tf.tanh(WHq+tf.expand_dims(G,1)) # N,Q,Hidden_size 
                a = tf.nn.softmax( tf.reduce_sum(G*Wt, 2) + Ba ) # N,Q+1
                # inputs
                Hqa = tf.reduce_sum( tf.expand_dims(a, -1) * hq_stack, 1) # N,Hidden_size   
                z = tf.concat( 1, [H_p[i], Hqa] ) 
                # lstm
                h, state = cell(z, state) # N,Hidden_size
                H.append(h)
        return H
    
    def build_model(self):

        self.passage = tf.placeholder(tf.int64, shape=[self._batch_size, self._p_length])
        self.question = tf.placeholder(tf.int64, shape=[self._batch_size, self._q_length])
        self.answers = tf.placeholder(tf.int64)

        with tf.variable_scope('embedding'):
            W_p = tf.get_variable('W_p', shape=[self._voca_size, self._embedding_size])
            W_q = tf.get_variable('W_q', shape=[self._voca_size, self._embedding_size])
            p = tf.nn.embedding_lookup(W_p, tf.transpose(self.passage), name='p_embed') # P,N,E 
            qs = tf.nn.embedding_lookup(W_q, tf.transpose(self.question), name='q_embed') # Q,N,E

        # as = tf.unpack(self.answers)
        with tf.variable_scope('preprocess'):
            # print 'p shape', p.get_shape()
            p = tf.unpack(p)
            qs = tf.unpack(qs)

            H_p, p_final_state = tf.nn.rnn(
                    tf.nn.rnn_cell.LSTMCell(self._hidden_size ), p,
                    dtype=p[0].dtype, scope='plstm')

            H_q, q_final_state = tf.nn.rnn(
                    tf.nn.rnn_cell.LSTMCell(self._hidden_size), qs,
                    dtype=qs[0].dtype, scope='qlstm')  # [N,Hidden]*T

        with tf.name_scope('match'):
            Wr = tf.get_variable('Wr', shape=[self._hidden_size, self._hidden_size])
            Bg = tf.get_variable('Bg', shape=[self._hidden_size])
            Wt = tf.get_variable('Wt', shape=[self._hidden_size])
            Ba = tf.get_variable('Ba', shape=[len(H_q)+1])
            FH = []; BH=[];

            # calculate WHq for all future use
            Wq = tf.get_variable('Wq', shape=[self._hidden_size, self._hidden_size])
            hq_stack = tf.pack(H_q, axis=1) 
            # append NULL
            shape = hq_stack.get_shape().as_list()
            shape[1] = 1
            hq_stack = tf.concat(1, [hq_stack, tf.zeros(shape, dtype=hq_stack.dtype)]) # N,Q+1,Hidden_size

            hq_shape = hq_stack.get_shape().as_list()
            hq_shape[0] = -1
            tmp = tf.reshape(hq_stack, [-1,self._hidden_size]) 
            WHq = tf.matmul(tmp, Wq) # N*(Q+1), Hidden_size
            WHq = tf.reshape(WHq, hq_shape) # N,Q+1,Hidden_size
            
            Wp = tf.get_variable('Wp', shape=[self._hidden_size, self._hidden_size])

            pass_param = [Wp, Wr, Bg, WHq, H_p, Wt, Ba, hq_stack]
            FH = self.match_unit('forward', pass_param)
            BH = self.match_unit('backward',pass_param)

            H = []
            for f,b in zip(FH,BH):
                H.append( tf.concat(1,[f,b] ))
            # append STOP
            STOP = tf.zeros_like(H[0])
            H.append(STOP)
            self.H=H

        # No Forloop in pointer, handle from outside
        with tf.name_scope('pointer'):
            # calculate VH for all future use
            V = tf.get_variable('V', shape=[2*self._hidden_size, self._hidden_size])
            H_ = tf.pack(self.H, 1) # N, P+1, 2*Hidden_size
            shape = H_.get_shape().as_list()
            shape[-1] = int(shape[-1]/2)
            H = tf.reshape( H, [-1, 2*self._hidden_size] )
            HV = tf.reshape( tf.matmul(H,V), shape ) # N, P+1, Hidden_size

            self.in_h = h = tf.placeholder(self.H[0].dtype)
            Wf = tf.get_variable('Wf', shape=[self._hidden_size, self._hidden_size])
            Bf = tf.get_variable('Bf', shape=[self._hidden_size])
            vt = tf.get_variable('vt', shape=[self._hidden_size])
            c  = tf.get_variable('c', shape=[len(self.H)])

            F  = tf.matmul(h,Wf)+Bf # N, Hidden_size
            F  = tf.tanh( HV+tf.expand_dims(F,1) ) # N,P+1,Hidden_size
            self.predict = beta = tf.nn.softmax( tf.reduce_sum( F*vt, 2 )+c ) # N, P+1

            # waiting for inference call
            self.in_state = state = [   tf.placeholder(self.H[0].dtype, shape=[self._batch_size, self._hidden_size]), 
                                        tf.placeholder(self.H[0].dtype, shape=[self._batch_size, self._hidden_size])]
            inputs = tf.reduce_sum(H_*tf.expand_dims(beta, 2), 2)
            self.pointer_cell = tf.nn.rnn_cell.BasicLSTMCell(self._hidden_size, state_is_tuple=True)
            self.out_h, self.out_state = self.pointer_cell(inputs, state)

    @property
    def pointer_zero_state(self):
        return self.pointer_cell.zero_state(self._batch_size, self.predict.dtype)

    def encode_step(self, sess, passage, question):
        return sess.run(fetch=self.H, 
                feed_dict={ self.passage: passage, self.question:question })

    def inference_step(self, sess, inputs, state):
        predict, state = sess.run( fetch=[self.predict, self.out_state],
                feed_dict={ self.in_state:state } )
        return predict, state, None