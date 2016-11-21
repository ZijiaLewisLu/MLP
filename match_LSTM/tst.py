# import tensorflow as tf
from match_LSTM import MatchLSTM

batch_size = 1
vocab_size = 82788
learning_rate=1e-2

model = MatchLSTM(100, 20, 15, batch_size, vocab_size)
model.build_model()
