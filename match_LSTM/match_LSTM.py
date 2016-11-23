import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from cells import Match_cell, Pointer_cell


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
                    embedding_size=128, hidden_size=128,
                    optim=tf.train.RMSPropOptimizer(1e-4, momentum=0.9, decay=0.95),
                    ):
        """
        vocab_size, question_length, passage_length, batch_size, embedding_size=128, hidden_size=128
        """
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.q_length = q_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.p_length = p_length
        self.a_length = a_length

        self.passage = None
        self.question = None
        self.answer = None
        
        self.optim = optim
        self.train_op = None

    # def match_unit(self, direction, param):
    #     Wp, Wr, Bg, WHq, H_p, Wt, Ba, hq_stack = param
    #     H = []
    #     with tf.variable_scope(direction) as scope:
    #         cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True)
    #         state = cell.zero_state(self.batch_size, hq_stack.dtype)
    #         h = state.h
    #         idx = range(len(H_p))
    #         if direction == 'backward': idx=list(reversed(idx))

    #         for i in idx:
    #             if i != idx[0] :
    #                 scope.reuse_variables()
    #             # attention
    #             G = tf.matmul(H_p[i],Wp)+tf.matmul(h, Wr) + Bg # N, Hidden_size
    #             G = tf.tanh(WHq+tf.expand_dims(G,1)) # N,Q,Hidden_size 
    #             a = tf.nn.softmax( tf.reduce_sum(G*Wt, 2) + Ba ) # N,Q+1
    #             # inputs
    #             Hqa = tf.reduce_sum( tf.expand_dims(a, -1) * hq_stack, 1) # N,Hidden_size   
    #             z = tf.concat( 1, [H_p[i], Hqa] ) 
    #             # lstm
    #             h, state = cell(z, state) # N,Hidden_size
    #             H.append(h)
    #     return H

    def build_model(self):
        self.passage = tf.placeholder(tf.int32, shape=[self.batch_size, self.p_length], name='passage')
        self.question = tf.placeholder(tf.int32, shape=[self.batch_size, self.q_length],name='question')
        self.answer = tf.placeholder(tf.int32, shape=[self.batch_size, self.a_length, self.p_length],name='answer')
        self.p_end = tf.placeholder(tf.int32, shape=[self.batch_size], name='p_end')
        self.q_end = tf.placeholder(tf.int32, shape=[self.batch_size], name='q_end')
        self.a_end = tf.placeholder(tf.int32, shape=[self.batch_size], name='a_end')

        with tf.variable_scope('embedding'):
            E_p = tf.get_variable('E_p', shape=[self.vocab_size, self.embedding_size])
            E_q = tf.get_variable('E_q', shape=[self.vocab_size, self.embedding_size])
            sep = tf.histogram_summary("E_p",E_p)
            seq = tf.histogram_summary('E_q',E_q)
            self.embed_sum = tf.merge_summary([sep, seq])
            p = tf.nn.embedding_lookup(E_p, self.passage, name='p_embed') # N,P,E 
            qs = tf.nn.embedding_lookup(E_q,self.question, name='q_embed') # N,Q,E

        with tf.variable_scope('preprocess'):
            # N,T,H
            H_p, p_final_state = tf.nn.dynamic_rnn(
                            rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True),
                            p, sequence_length=self.p_end, dtype=tf.float32, scope='plstm')

            H_q, q_final_state = tf.nn.dynamic_rnn(
                            rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True), 
                            qs, sequence_length=self.q_end, dtype=tf.float32, scope='qlstm')

        with tf.variable_scope('match'):
            Wr = tf.get_variable('Wr', shape=[self.hidden_size, self.hidden_size])
            Bg = tf.get_variable('Bg', shape=[self.hidden_size])
            Wt = tf.get_variable('Wt', shape=[self.hidden_size])
            Ba = tf.get_variable('Ba', shape=[self.q_length])

            # calculate WHq for all future use
            Wq = tf.get_variable('Wq', shape=[self.hidden_size, self.hidden_size])
            tmp = tf.reshape(H_q, [-1, self.hidden_size]) 
            WHq = tf.matmul(tmp, Wq) # N,Q,Hidden_size
            WHq = tf.reshape(WHq, [self.batch_size, self.q_length, self.hidden_size]) # N,Q,Hidden_size
            
            Wp = tf.get_variable('Wp', shape=[self.hidden_size, self.hidden_size])

            share_param = [Wp, Wr, Bg, WHq, Wt, Ba, H_q]
            H, p_final_state, = tf.nn.bidirectional_dynamic_rnn(
                Match_cell(self.hidden_size, share_param, state_is_tuple=True),
                Match_cell(self.hidden_size, share_param, state_is_tuple=True),
                H_p, sequence_length=self.p_end, dtype=tf.float32, scope='match')

            # N,P,2*H
            self.H = tf.concat(2,H)
            # print 'H', self.H.get_shape()
            # import ipdb
            # ipdb.set_trace()

        with tf.variable_scope('pointer'):
            # calculate VH for all future use
            V = tf.get_variable('V', shape=[2*self.hidden_size, self.hidden_size])
            H_ = tf.reshape( self.H, [-1, 2*self.hidden_size] )
            HV = tf.reshape( tf.matmul(H_,V), [self.batch_size, self.p_length, self.hidden_size], name='HV') # N, P, H

            Wf = tf.get_variable('Wf', shape=[self.hidden_size, self.hidden_size])
            Bf = tf.get_variable('Bf', shape=[self.hidden_size])
            vt = tf.get_variable('vt', shape=[self.hidden_size])
            c  = tf.get_variable('c', shape=[self.p_length])
            share_param = [Wf, Bf, HV, vt, c, self.H]

            pointer_cell = Pointer_cell(self.hidden_size, share_param, state_is_tuple=True)
            # tf.nn.dynamic_rnn( pointer_cell, self.answer, sequence_length=self.a_length, dtype=tf.float32, scope='lstm')
            tf.nn.rnn( pointer_cell, tf.unpack(self.answer, axis=1), dtype=tf.float32, scope='lstm')

        epsilon = tf.constant(value=0.00001, shape=[self.p_length])
        self.score = tf.pack(pointer_cell.score, 1, name='score') + epsilon # N,A,P
        # logit = tf.nn.log_softmax(self.score, name='logit')
        # self.answer = tf.cast(self.answer, tf.float32)
        # self.loss = -logit*self.answer

        base = tf.cast(tf.reduce_sum(self.a_end),"float")
        self.loss = tf.nn.softmax_cross_entropy_with_logits( self.score, self.answer, name='loss')
        self.loss_sum  = tf.scalar_summary("loss", tf.reduce_sum(self.loss)/base)
        self.vloss_sum  = tf.scalar_summary("V_loss", tf.reduce_sum(self.loss)/base)


        # self.score = tf.nn.softmax(self.score, name='softmax') 
        prediction = tf.argmax(self.score,2,name='prediction')
        Y = tf.argmax(self.answer,2)
        correct_predictions = tf.equal(prediction, Y)
        mask = tf.greater(Y, tf.zeros_like(Y), name='accuracy_mask')
        correct_predictions = tf.logical_and(mask, correct_predictions)
        self.accuracy = tf.reduce_sum(tf.cast(correct_predictions, "float")) / base 
        self.acc_sum   = tf.scalar_summary("accuracy", self.accuracy)
        self.vacc_sum   = tf.scalar_summary("V_accuracy", self.accuracy)

        # self.loss = tf.nn.softmax_cross_entropy_with_logits(self.score, self.answer, name='softmax_loss')
        # self.loss = -tf.reduce_sum(self.answer*tf.log(self.score))
        # y = tf.sparse_tensor_to_dense(self.answer)
        # y = self.answer
        # y_ = tf.log(self.score)
        # print y.dtype, y_.dtype
        # print y.get_shape().as_list(),  y_.get_shape().as_list(), self.score.get_shape().as_list()
        # yy_ = y*y_

        self.grads_and_vars = self.optim.compute_gradients(self.loss)
        checker = []
        for g, v in self.grads_and_vars:
            name = v.name
            checker.append( tf.check_numerics(g, "variable:%s"%name) )
            if g is not None: checker.append( tf.check_numerics(g, "gradient:%s"%name) )
        self.check_op = tf.group(*checker)
        self.train_op = self.optim.apply_gradients(self.grads_and_vars)

        self.gv_sum = self.contruct_summaries()
        self.train_sum = tf.merge_summary([self.loss_sum, self.acc_sum, self.gv_sum])
        self.validate_sum = tf.merge_summary([self.vloss_sum, self.vacc_sum, self.embed_sum])
    
    def contruct_summaries(self):
        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                gs = tf.scalar_summary("I_{}-grad".format(v.name), tf.reduce_mean(g))
                vs = tf.scalar_summary("I_{}-var".format(v.name), tf.reduce_mean(v))
                grad_summaries.append(gs)
                grad_summaries.append(vs)
        grad_summaries_merged = tf.merge_summary(grad_summaries)
        # predict_summaries = tf.histogram_summary("prediction", self.prediction)
        # self.summaries = tf.merge_summary([grad_summaries_merged])
        return grad_summaries_merged

    def step(self, sess, passage, question, answer, train=True):
        """
        Input: session, passage, question, answer
        Output: 
            if train==True: predictions, accuracy, loss
            else: predictions
        """
        if train:
            # print '.....\n', answer
            assert self.train_op is not None
            result = sess.run(
                fetches=[self.train_op, self.score, self.loss,],
                feed_dict={ self.passage: passage, self.question: question, self.answer:answer})
            result = result[1:]

        else:
            result = sess.run(
                fetches=[self.prediction],
                feed_dict={ self.passage:passage, self.question:question})
        return result
