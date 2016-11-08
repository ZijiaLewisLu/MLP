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

class MatchLSTM():
    
    def __init__(self, p_length, q_length, a_length, batch_size, vocab_size,
                    embedding_size=128, hidden_size=128):
        """
        vocab_size, question_length, passage_length, batch_size, embedding_size=128, hidden_size=128
        """
        self._embedding_size = embedding_size
        self._hidden_size = hidden_size
        self._q_length = q_length
        self._voca_size = vocab_size
        self._batch_size = batch_size
        self._p_length = p_length
        self._a_length = a_length

        self.passage = None
        self.question = None
        self.answer = None

        self.optim = None
        self.train_op = None

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
    
    def assign_optim(self, optim):
        self.optim = optim

    def build_model(self):
        assert self.optim is not None

        self.passage = tf.placeholder(tf.int64, shape=[self._batch_size, self._p_length])
        self.question = tf.placeholder(tf.int64, shape=[self._batch_size, self._q_length])
        self.answer = tf.placeholder(tf.int64, shape=[self._batch_size, self._a_length])

        with tf.variable_scope('embedding'):
            E_p = tf.get_variable('E_p', shape=[self._voca_size, self._embedding_size])
            E_q = tf.get_variable('E_q', shape=[self._voca_size, self._embedding_size])
            p = tf.nn.embedding_lookup(E_p, tf.transpose(self.passage), name='p_embed') # P,N,E 
            qs = tf.nn.embedding_lookup(E_q, tf.transpose(self.question), name='q_embed') # Q,N,E

        with tf.variable_scope('preprocess'):
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

        with tf.variable_scope('pointer') as scope:
            # calculate VH for all future use
            V = tf.get_variable('V', shape=[2*self._hidden_size, self._hidden_size])
            H_ = tf.pack(self.H, 1) # N, P+1, 2*Hidden_size
            shape = H_.get_shape().as_list()
            shape[-1] = int(shape[-1]/2)
            H = tf.reshape( H, [-1, 2*self._hidden_size] )
            HV = tf.reshape( tf.matmul(H,V), shape ) # N, P+1, Hidden_size

            Wf = tf.get_variable('Wf', shape=[self._hidden_size, self._hidden_size])
            Bf = tf.get_variable('Bf', shape=[self._hidden_size])
            vt = tf.get_variable('vt', shape=[self._hidden_size])
            c  = tf.get_variable('c', shape=[len(self.H)])
            self.pointer_cell = tf.nn.rnn_cell.BasicLSTMCell(self._hidden_size, state_is_tuple=True)
            state = self.pointer_cell.zero_state(self._batch_size, H.dtype)
            h = state.h
            score = []
            # self.predict = []
            for step in range(self._a_length):
                if step>0: scope.reuse_variables()
                F  = tf.matmul(h,Wf)+Bf # N, Hidden_size
                F  = tf.tanh( HV+tf.expand_dims(F,1) ) # N,P+1,Hidden_size
                beta = tf.nn.softmax( tf.reduce_sum( F*vt, 2 )+c ) # N,P+1
                score.append(beta)

                inputs = tf.reduce_sum(H_*tf.expand_dims(beta, 2), 2)
                h, state = self.pointer_cell(inputs, state)
        
        self.score = tf.pack(score,1) #N,A,P+1
        self.prediction = tf.argmax(self.score,2, name='prediction')
        
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.score, self.answer, name='softmax_loss')
        self.loss = tf.reduce_mean(self.loss)

        correct_predictions = tf.equal(self.prediction, self.answer)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        self.train_op = self.optim.minimize(self.loss)
    
    @property
    def pointer_zero_state(self):
        return self.pointer_cell.zero_state(self._batch_size, self.predict.dtype)

    def step(self, sess, passage, question, answer, train=True):
        """
        Input: session, passage, question, answer
        Output: 
        if train==True: predictions, accuracy, loss
        else: predictions
        """
        if train:
            assert self.train_op is not None
            result = sess.run(
                fetches=[self.train_op, self.prediction, self.accuracy, self.loss],
                feed_dict={ self.passage: passage, self.question: question, self.answer:answer})
            result = result[1:]

        else:
            result = sess.run(
                fetches=[self.prediction],
                feed_dict={ self.passage:passage, self.question:question})
        return result