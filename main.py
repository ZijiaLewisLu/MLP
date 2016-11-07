import tensorflow as tf
from match_LSTM import MatchLSTM
import data_utils as D
import json

batch_size = 10
vocab_size = 99036

# prepare data
with open('./squad/train-v1.1.json') as f:
    squad = json.load(f)
pair, _ = D.format_data(squad)
p,q,a = D.transform('./squad/vocab_99036.json', pair)

# build model
g = tf.Graph()
sess = tf.Session()
q_length = q.shape[1]
p_length = p.shape[1]
print q_length, p_length
model = MatchLSTM(vocab_size,p_length, q_length, batch_size)
model.build_model()

# forward once
init = tf.initialize_all_variables()
sess.run(init)
print 'Start Forwarding'
answer = sess.run( model.answer, 
                    feed_dict={ model.passage: p[:batch_size], model.question:q[:batch_size] } )
print answer.shape
print answer