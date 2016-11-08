import tensorflow as tf
import sys
from match_LSTM import MatchLSTM
import data_utils as D
# import json

num_epoch = 20
batch_size = 10
vocab_size = 82788
small_size = 50
learning_rate = 1e-2

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
model = MatchLSTM(p_length, q_length, a_length, batch_size, vocab_size)

model.assign_optim(tf.train.GradientDescentOptimizer(learning_rate))

model.build_model()
init = tf.initialize_all_variables()
sess.run(init)
print 'Sir, model inited'
sys.stdout.flush()

for e in range(num_epoch):
    bi = D.batchIter(batch_size, [p, q, a])
    for b, batch in enumerate(bi):
        batch.insert(0, sess)
        pre, acc, loss = model.step(*batch)
        print 'E%d B%d acc %0.3f loss %0.3f' % (e, b, acc, loss)
