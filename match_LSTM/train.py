#! /usr/bin/python
import tensorflow as tf
import sys
from match_LSTM import MatchLSTM
import data_utils as D
import numpy as np
import time
from datetime import timedelta
# import json


tf.flags.DEFINE_boolean("test", False, "No Backward")
tf.flags.DEFINE_string('logto', './run', 'LogDir')
tf.flags.DEFINE_integer('epoch', 50, 'num of epoch')
tf.flags.DEFINE_integer('vocab',82788, 'vocab size')
# Misc Parameters

FLAGS = tf.flags.FLAGS

if FLAGS.test:
    num_epoch = 1
else:
    num_epoch = FLAGS.epoch

batch_size = 10
vocab_size = 82788
small_size = 50
learning_rate = 1e-1

# prepare data
p, q, a = D.load_data()
if small_size is not None:
    p = p[:small_size]
    q = q[:small_size]
    a = a[:small_size]
p_length = p.shape[1]
q_length = q.shape[1]
a_length = a.shape[1]
print 'Sir, data are ready!'
print p_length, p_length, a_length
sys.stdout.flush()

# build model
g = tf.Graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
model = MatchLSTM(p_length, q_length, a_length, batch_size, vocab_size, learning_rate=learning_rate)

start = time.time()
model.build_model()
end = time.time()
print "Build Model use:", str(timedelta(end - start))

# summary
writer = tf.train.SummaryWriter(FLAGS.logto, sess.graph)
grad_summaries = []
for g, v in model.grads_and_vars:
    if g is not None:
        grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
grad_summaries_merged = tf.merge_summary(grad_summaries)
predict_summaries = tf.histogram_summary("prediction", model.prediction)
summaries = tf.merge_summary([grad_summaries_merged, predict_summaries])

init = tf.initialize_all_variables()
start = time.time()
sess.run(init)
end = time.time()
print "Init Variable use:", str(timedelta(end - start))

print 'Sir, model inited'
sys.stdout.flush()

batch_num = p.shape[0]/batch_size + 1


def one_step(P, Q, A, show_gv=True):
    start = time.time()
    # pre, loss = model.step(sess, P,Q,A)
    _, pre, loss, gv, summ = sess.run(
        fetches=[model.train_op, model.score, model.loss, model.grads_and_vars, summaries],
        feed_dict={ model.passage: P, model.question: Q, model.answer:A})
    # result = result[1:]
    end = time.time()
    print timedelta(seconds=(end-start))
    print 'loss',loss
    p = np.argmax(pre,axis=2)
    y = np.argmax(A,axis=2)    
    correct = ( p == y )
    print 
    print p,y
    print 'acc', np.mean(correct)
    print 
    
    if show_gv:
        for g, v in gv:
            if g is not None:
                if isinstance(g, np.ndarray):
                    print g.shape, np.linalg.norm(v), np.max(g)
                else:
                    print g.dense_shape, np.linalg.norm(v), np.max(g.values)
    print '\n\n\n'
    writer.add_summary(summ)
    return (pre, loss, gv)

for e in range(num_epoch):
    bi = D.batchIter(batch_size, [p, q, a])
    A = 0
    L = 0
    for b, batch in enumerate(bi):
        # batch.insert(0, sess)

        # transform answer
        answer = batch[-1] # batch_size, aLen+1
        shape = [ batch_size, a_length, p_length+1 ]
        new = np.zeros(shape)
        # index = []
        for i, a_ in enumerate(answer):
            # new = [ [i,t,loc] for t, loc in enumerate(a) if loc != 0 ]
            # index.extend(new)
            for t, loc in enumerate(a_):
                if loc!=0:
                    new[i,t,loc] = 1
        batch[-1] = new
        # value = [1]*len(index)
        # sp = tf.SparseTensor(indices=index, values=value, shape=shape)
        # sp_value = sp.eval(session=sess)
        # print batch[-1] 

        p_, q_, a_ = batch

        pre, loss, gv = one_step(p_,q_,a_)

        # Forward and Backward
        # pre, loss = model.step(*batch)

        # _, score, loss, summ = sess.run( 
        #     fetches=[model.train_op, model.score, model.loss, model.grads_and_vars, summaries],
        #     feed_dict={model.passage:p_, model.question:q_, model.answer:a_} )

        # writer.add_summary(summ)
        
        # L += loss
        # # print pre.shape
        # predict = np.argmax(score, axis=2)
        # # print predict.shape
        # print predict[:3]
        # print '^P___________________________________________|A'
        # print a[:3]
        # correct_prediction = (predict==answer)
        # # print correct_prediction
        # acc = np.mean(correct_prediction)
        # print acc
        # A += acc
        # print 'max, min of prediction', np.max(pre), np.min(pre)
        # print 'E%d B%d acc %0.3f loss %0.3f' % (e, b, acc, loss)
    # print 'E%d, acc %0.3f loss %0.6f' %( e, A/batch_num, L/batch_num)
