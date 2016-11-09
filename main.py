import tensorflow as tf
import sys
from match_LSTM import MatchLSTM
import data_utils as D
import numpy as np
# import json

num_epoch = 20
batch_size = 10
vocab_size = 82788
small_size = 50
learning_rate = 1e-1
summary_dir = './runs'

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
sess = tf.Session()
model = MatchLSTM(p_length, q_length, a_length, batch_size, vocab_size, learning_rate=learning_rate, test=False)
model.build_model()

# summary
writer = tf.train.SummaryWriter(summary_dir, sess.graph)
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
sess.run(init)
print 'Sir, model inited'
sys.stdout.flush()

batch_num = p.shape[0]/batch_size + 1


for e in range(num_epoch):
    bi = D.batchIter(batch_size, [p, q, a])
    A = 0
    L = 0
    for b, batch in enumerate(bi):
        batch.insert(0, sess)
        
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

        _, p , q, a = batch

        # Forward and Backward
        # pre, loss = model.step(*batch)

        _, score, loss, summ = sess.run( 
            fetches=[model.train_op, model.score, model.loss, model.grads_and_vars, summaries],
            feed_dict= {model.passage:p, model.question:q, model.answer:a} )

        writer.add_summary(summ)
        
        L += loss
        # print pre.shape
        predict = np.argmax(score, axis=2)
        # print predict.shape
        print predict[:3]
        print '^P___________________________________________|A'
        print a[:3]
        correct_prediction = (predict==answer)
        # print correct_prediction
        acc = np.mean(correct_prediction)
        print acc
        A += acc
        # print 'max, min of prediction', np.max(pre), np.min(pre)
        # print 'E%d B%d acc %0.3f loss %0.3f' % (e, b, acc, loss)
    print 'E%d, acc %0.3f loss %0.6f' %( e, A/batch_num, L/batch_num)