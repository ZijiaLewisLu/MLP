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
    
    def __init__(self, vocab_size, q_length, embedding_size=128, hidden_size=128):
        self._embedding_size = embedding_size
        self._hidden_size = hidden_size
        self._q_length = q_length
        self._voca_size = vocab_size

    def build_model(self):
        self.passage = tf.placeholder(tf.int64, shape=[None])
        self.question = tf.placeholder(tf.int64, shape=[None, self._q_length])
        self.answers = tf.placeholder(tf.int64)

        with tf.name_scope('embedding'):
            W_p = tf.get_variable('W_p', shape=[self._voca_size, self._embedding_size])
            W_q = tf.get_variable('W_q', shape=[self._voca_size, self._embedding_size])
            p = tf.nn.embedding_lookup(W_p, self.passage, name='p_embed')
            qs = tf.nn.embedding_lookup(W_q, tf.transpose(self.question), name='q_embed') # T, N, E
	    # print qs.dtype

        # as = tf.unpack(self.answers)
        with tf.name_scope('preprocess'):
            p = [p]
            qs = tf.unpack(qs)

            H_p, p_final_state = tf.nn.rnn(
                    tf.nn.rnn_cell.LSTMCell(self._hidden_size ), p,
                    dtype=p[0].dtype, scope='plstm')
            print H_p[0].name

            H_q, q_final_state = tf.nn.rnn(
                    tf.nn.rnn_cell.LSTMCell(self._hidden_size), qs,
                    dtype=qs[0].dtype, scope='qlstm')  # [N,Hidden]*T

        with tf.name_scope('match'):
            Wr = tf.get_variable('Wr', shape=[self._hidden_size, self._hidden_size])
            Bg = tf.get_variable('Bg', shape=[self._hidden_size])
            Wt = tf.get_variable('Wt', shape=[self._hidden_size])
            Ba = tf.get_variable('Wa', shape=[len(H_q)])
            FH = []; BH=[];

            # calculate WHq for all future use
            Wq = tf.get_variable('Wq', shape=[self._hidden_size, self._hidden_size])
            hq_stack = tf.pack(H_q, axis=1) # N,Q,Hidden_size
            hq_shape = hq_stack.get_shape().as_list()
            hq_shape[0] = -1
            tmp = tf.reshape(hq_stack, [-1,self._hidden_size]) 
            WHq = tf.matmul(tmp, Wq) # N*Q, Hidden_size
            # print WHq.get_shape()
            # print hq_shape
            WHq = tf.reshape(WHq, hq_shape) # N,Q,Hidden_size
            # append NULL
            shape = WHq.get_shape().as_list()
            shape[1]=1 # N,1,Hidden_size 
            NULL = tf.zeros(shape, dtype=WHq.dtype)
            WHq = tf.concat(1, [WHq, NULL])
            
            # calculate WHp for all future use
            Wp = tf.get_variable('Wp', shape=[self._hidden_size, self._hidden_size])
            # WHp = []
            # for p in H_p:
                # WHp.append( tf.matmul(p, Wp))

            with tf.name_scope('forward'):
                cell = tf.nn.rnn_cell.BasicLSTMCell(self._hidden_size, state_is_tuple=True)
                batch_size = hq_stack.get_shape().as_list()[0]
                state = cell.zero_state(batch_size, H_q.dtype)
                h = state.h
                for i in range(len(H_p)):
                    # attention
                    G = tf.matmul(H_p[i],Wp)+tf.matmul(h, Wr) + Bg # Hidden_size
                    G = tf.tanh(WHq+G) # N,Q,Hidden_size 
                    a = tf.nn.softmax( tf.reduce_sum(G*Wt, 2) + Ba ) # N,Q
                    # inputs
                    Hqa = tf.reduce_sum( tf.expand_dims(a, -1) * hq_stack, 1) # N,Hidden_size   
                    Hpi = tf.tile( tf.expand_dims(H_p[i],0), [batch_size,1]) # N,Hidden_size
                    z = tf.concat( 1, [Hpi, Hqa] ) 
                    # lstm
                    h, state = cell(z, state) # N,Hidden_size
                    FH.append(h)

            with tf.name_scope('backward'):
                cell = tf.nn.rnn_cell.BasicLSTMCell(self._hidden_size, state_is_tuple=True)
                batch_size = hq_stack.get_shape().as_list()[0]
                state = cell.zero_state(batch_size, H_q.dtype)
                h = state.h
                for i in reversed(range(len(H_p))):
                    # attention
                    G = tf.matmul(H_p[i],Wp)+tf.matmul(h, Wr) + Bg # Hidden_size
                    G = tf.tanh(WHq+G) # N,Q,Hidden_size 
                    a = tf.nn.softmax( tf.reduce_sum(G*Wt, 2) + Ba ) # N,Q
                    # inputs
                    Hqa = tf.reduce_sum( tf.expand_dims(a, -1) * hq_stack, 1) # N,Hidden_size   
                    Hpi = tf.tile( tf.expand_dims(H_p[i],0), [batch_size,1]) # N,Hidden_size
                    z = tf.concat( 1, [Hpi, Hqa] ) 
                    # lstm
                    h, state = cell(z, state) # N,Hidden_size
                    BH.append(h)

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
            H = tf.reshape( H, [-1, 2*self._hidden_size] )
            HV = tf.reshape( tf.matmul(H,V), shape ) # N, P+1, Hidden_size

            self.in_h = h = tf.placeholder(self.H[0].dtype)
            Wa = tf.get_variable('Wa', shape=[self._hidden_size, self._hidden_size])
            Ba = tf.get_variable('Ba', shape=[self._hidden_size])
            vt = tf.get_variable('vt', shape=[self._hidden_size])
            c  = tf.get_variable('c', shape=[len(self.H)])

            F  = tf.matmul(h,Wa)+Ba # N, Hidden_size
            F  = tf.tanh( HV+tf.expand_dims(F,1) ) # N,Q+1,Hidden_size
            self.predict = beta = tf.nn.softmax( tf.reduce_sum( F*vt, 2 )+c )

            # waiting for inference call
            self.in_state = state = tf.placeholder(self.H[0].dtype)
            input = tf.reduce_sum(H_*tf.expand_dims(beta, 1))
            self.pcell = tf.nn.rnn_cell.BasicLSTMCell(self._hidden_size, name='LSTM', state_is_tuple=True)
            self.out_h, self.out_state = self.pcell(input, state)

    def encode_step(self, sess, passage, question):
        return sess.run(fetch=self.H, 
                feed_dict={ self.passage: passage, self.question:question })

    def inference_step(self, sess, h, state):
        return sess.run( fetch=[self.predict, self.out_h, self.out_state],
                feed_dict={ self.in_h:h, self.in_state:state } )

