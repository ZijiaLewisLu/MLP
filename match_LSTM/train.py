#! /usr/bin/python
import tensorflow as tf
import sys
from match_LSTM import MatchLSTM
import data_utils as D
import numpy as np
import time
from datetime import timedelta
import os
# import json


tf.flags.DEFINE_boolean("test", False, "No Backward")
tf.flags.DEFINE_string('logto', './run', 'LogDir')
tf.flags.DEFINE_integer('epoch', 50, 'num of epoch')
tf.flags.DEFINE_integer('vocab',50000, 'vocab size')
FLAGS = tf.flags.FLAGS

if FLAGS.test:
    num_epoch = 1
else:
    num_epoch = FLAGS.epoch
batch_size = 10
vocab_size = 50000
small_size = 1000
learning_rate = 1e-1
pLen=300
qLen=20 
aLen=10 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


# # prepare data
# print 'Sir, data are ready!'
# sys.stdout.flush()

# build model
# g = tf.Graph()
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
sess = tf.Session()
optim=tf.train.RMSPropOptimizer(learning_rate, momentum=0.9, decay=0.95)
model = MatchLSTM(pLen, qLen, aLen, batch_size, vocab_size, optim=optim)

start = time.time()
model.build_model()
end = time.time()
print "Build Model use:", str(end - start)

log_dir = "%s/%s"%(FLAGS.logto, time.strftime("%m_%d_%H_%M"))
save_dir = os.path.join(log_dir, 'ckpts')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    # with open(log_dir+'/Flags.js','w') as f:
    #   json.dump(FLAGS.__flags, f, indent=4)
else:
    print 'log_dir exist %s' % log_dir
    exit(2)

# summary
writer = tf.train.SummaryWriter(log_dir, sess.graph)
saver = tf.train.Saver()

init = tf.initialize_all_variables()
start = time.time()
sess.run(init)
end = time.time()
print "Sir, model inited. Use:", end - start
sys.stdout.flush()

counter = 0
start_time = time.time()
for e in range(num_epoch):
    titer, tstep, viter, vstep = D.load_data(pLen=pLen, aLen=aLen, qLen=qLen, batch_size=batch_size, data_size=small_size, vocab_size=vocab_size)
    running_loss = 0.0
    running_acc  = 0.0
    for idx, P, p_end, Q, q_end, A, a_end in titer:
        feed = { 
            model.passage: P,
            model.question: Q,
            model.answer: A,
            model.p_end: p_end,
            model.q_end: q_end,
            model.a_end: a_end,
        }

        fetch = [ model.train_op, model.check_op, model.accuracy, model.loss, model.train_sum]

        _, _, acc, loss, summary = sess.run( fetch, feed )

        writer.add_summary(summary, counter)
        running_acc += acc
        running_loss += np.mean(loss)
        if counter % 10 == 0:
            print "Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f" \
                % (e, idx, tstep, time.time() - start_time, running_loss/10.0, running_acc/10.0)
            running_loss = 0
            running_acc = 0
        counter += 1

        if (counter+1) % 1000 == 0:
            saver.save(sess, os.path.join(save_dir, 'model'), global_step=counter)


    running_acc  = 0.0
    running_loss = 0.0
    for idx, P, p_end, Q, q_end, A, a_end in viter:
        feed = { 
            model.passage: P,
            model.question: Q,
            model.answer: A,
            model.p_end: p_end,
            model.q_end: q_end,
            model.a_end: a_end,
        }

        fetch = [model.check_op, model.accuracy, model.loss]

        _, acc, loss = sess.run( fetch, feed )
        
        running_acc += acc
        running_loss += np.mean(loss)
        print "Epoch: [%2d] Validation time: %4.4f, loss: %.8f, accuracy: %.8f" \
                            %(e, time.time()-start_time, running_loss/vstep, running_acc/vstep)