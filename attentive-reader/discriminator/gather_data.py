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



DATA = []
LABEL = []
DLEN = []
fetch = [M.accuracy, logit, M.attention]
print 'Start!'
for data in titer:
    idx = data[0]
    doc = data[1]
    ans = data[-1]
    d_len = data[2]

    sys.stdout.write( '\r%d' % idx )
    # if idx == 300:
    #    break

    acc, prob, atten = step(data, M, sess, fetch)
    p, c, both = analyse(doc, ans, atten)

    doc = doc[both]
    atten = atten[both]
    d_len = d_len[both]

    tmp = np.stack( [doc, atten], axis=2 )
    DATA.append(tmp)
    DLEN.append(d_len)

    c = c[both]
    p = p[both]
    c = c.astype(np.int)
    p = p.astype(np.int)
    c[ c == 1 ] = 2
    label = c + p
    LABEL.append(label)

# print len(DATA)

d = np.concatenate( DATA )
l = np.concatenate( LABEL )
dl = np.concatenate( DLEN )

print d.shape
print l.shape
print dl.shape

with h5py.File('Data_Big.h5', 'w') as hf:
    hf.create_dataset('data', data=d)
    hf.create_dataset('dlen', data=dl)
    hf.create_dataset('label', data=l)
