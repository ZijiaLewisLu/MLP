import os
import tensorflow as tf
import time
import json

from model import DeepLSTM, DeepBiLSTM, AttentiveReader

from utils import pp, define_gpu

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [40]")
flags.DEFINE_integer("vocab_size", 50003, "The size of vocabulary [10000]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
flags.DEFINE_integer("gpu", 1, "the number of gpus to use")
flags.DEFINE_integer("data_size", 8000, "Number of files to train on")
flags.DEFINE_integer("hidden_size", 268, "Number of files to train on")
flags.DEFINE_float("learning_rate", 5e-5, "Learning rate [0.00005]")
flags.DEFINE_float("momentum", 0.9, "Momentum of RMSProp [0.9]")
flags.DEFINE_float("decay", 0.95, "Decay of RMSProp [0.95]")
flags.DEFINE_float("dropout", 1.0 , "Dropout rate")
flags.DEFINE_string("model", "Attentive", "The type of model to train and test [LSTM, BiLSTM, Attentive, Impatient]")
flags.DEFINE_string("data_dir", "data", "The name of data directory [data]")
flags.DEFINE_string("dataset", "cnn", "The name of dataset [cnn, dailymail]")
flags.DEFINE_string("log_dir", "log", "Directory name to save the log [log]")
flags.DEFINE_string("load_path", None, "The path to old model. [None]")
flags.DEFINE_string("optim", 'SGD', "The path to old model. [None]")
flags.DEFINE_string("active", 'relu', "The path to old model. [None]")

flags.DEFINE_boolean("forward_only", False, "True for forward only, False for training [False]")
FLAGS = flags.FLAGS

model_dict = {
  'LSTM': DeepLSTM,
  'BiLSTM': DeepBiLSTM,
  'Attentive': AttentiveReader,
  'Impatient': None,
}

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.gpu is not None:
    gpu_list  = define_gpu(FLAGS.gpu)

  log_dir = "%s/%s_%s"%(FLAGS.log_dir, time.strftime("%m_%d_%H_%M"), FLAGS.model)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    with open(log_dir+'/Flags.js','w') as f:
      json.dump(FLAGS.__flags, f, indent=4)
  else:
    print('log_dir exist %s' % log_dir)
    exit(2)


  with tf.Session() as sess:
    model = model_dict[FLAGS.model](batch_size=FLAGS.batch_size, dropout_rate=FLAGS.dropout,
                                    momentum=FLAGS.momentum, decay=FLAGS.decay,
                                    size=FLAGS.hidden_size,
                                    use_optimizer=FLAGS.optim,
                                    activitation=FLAGS.active)
    print " [*] Using GPU: ", gpu_list

    if not FLAGS.forward_only:
      model.train(sess, FLAGS.vocab_size, FLAGS.epoch,
                  FLAGS.data_dir, FLAGS.dataset, log_dir, FLAGS.load_path,
                  FLAGS.data_size)
    else:
      model.load(sess, FLAGS.checkpoint_dir, FLAGS.dataset)

if __name__ == '__main__':
  tf.app.run()
