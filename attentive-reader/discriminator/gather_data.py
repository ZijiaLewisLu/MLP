import os
import sys
sys.path.insert(0, '..')
import numpy as np
# import attentive_model, utils
# from attentive_model import AttentiveReader
import json
import tensorflow as tf
from utils import load_dataset
from evaluate import dig_tensors, step, analyse
# import pickle as pk
import h5py
# from sklearn.manifold import TSNE
# from collections import Counter


log_path = sys.argv[1]
if len(sys.argv) == 3:
    gpu = sys.argv[2]
else:
    gpu = 2

os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)


# log_path = './log/concat_B/12_10_02_29/'
ckpt = os.path.join(log_path, 'ckpts')
flags = os.path.join(log_path, 'Flags.js')
with open(flags, 'r') as f:
    old_flag = json.load(f)

for k in old_flag: print '%s: %s'%( k, old_flag[k])

max_nsteps=1000
max_query_length=20
batch_size = old_flag['batch_size']
vocab_size = old_flag['vocab_size']
hidden_size = old_flag['hidden_size']
activation = old_flag['activation']
atten = old_flag.get('attention', 'concat')

ck = tf.train.get_checkpoint_state(ckpt)
ckfiles = list(ck.all_model_checkpoint_paths)

tf.reset_default_graph()
g = tf.get_default_graph()
sess = tf.Session()

titer, tstep, viter, vstep = load_dataset('../data', 'cnn', 
                vocab_size, batch_size, max_nsteps, max_query_length, shuffle_data=False)

if tstep == 0:
    import ipdb; ipdb.set_trace()

which = -1
saver = tf.train.import_meta_graph(ckfiles[which]+'.meta')
saver.restore(sess, ckfiles[which])
M = dig_tensors(g)
logit = tf.nn.softmax(M.score)
print 'Load!'



DOC = []
DLEN = []
QUE = []
QLEN = []
LABEL = []

fetch = [M.accuracy, logit, M.attention]
print 'Start!'
for data in titer:
    idx = data[0]
    doc = data[1]
    ans = data[-1]
    d_len = data[2]
    query = data[3]
    q_len = data[4]

    sys.stdout.write( '\r%d' % idx )
    # if idx == 300:
    #    break

    acc, prob, atten = step(data, M, sess, fetch)
    p, c, both = analyse(doc, ans, atten)

    doc = doc[both]
    d_len = d_len[both]
    query = query[both]
    q_len = q_len[both]
    atten = atten[both]

    # tmp = np.stack( [doc, atten], axis=2 )
    DOC.append(doc)
    DLEN.append(d_len)
    QUE.append(query)
    QLEN.append(q_len)

    c = c[both]
    p = p[both]
    c = c.astype(np.int)
    p = p.astype(np.int)
    c[ c == 1 ] = 2
    label = c + p
    LABEL.append(label)

doc_all = np.concatenate( DOC )
dlen_all = np.concatenate( DLEN )
que_all = np.concatenate( QUE )
qlen_all = np.concatenate( QLEN )
label_all = np.concatenate( LABEL )

print doc_all.shape
print label_all.shape
print dlen_all.shape

with h5py.File('FULL_Big_BOOST.h5', 'w') as hf:
    hf.create_dataset('doc', data=doc_all)
    hf.create_dataset('dlen', data=dlen_all)
    hf.create_dataset('que', data=que_all)
    hf.create_dataset('qlen', data=qlen_all)
    hf.create_dataset('label', data=label_all)
