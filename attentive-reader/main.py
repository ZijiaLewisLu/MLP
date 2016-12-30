#! /usr/bin/python
import os
import tensorflow as tf
import time
import json
from utils import pp, define_gpu

flags = tf.app.flags
flags.DEFINE_integer("epoch", 15, "Epoch to train [40]")
flags.DEFINE_integer("vocab_size", 50003, "The size of vocabulary [10000]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [32]")
flags.DEFINE_integer("gpu", 1, "the number of gpus to use")
flags.DEFINE_integer("data_size", None, "Number of files to train on")
flags.DEFINE_integer("hidden_size", 256, "Hidden dimension for rnn and fully connected layer")
flags.DEFINE_integer("eval_every", 1000, "Eval every step")
flags.DEFINE_integer("D", 25, 'local_attention window size')
flags.DEFINE_float("learning_rate", 5e-5, "Learning rate [0.00005]")
flags.DEFINE_float("momentum", 0.9, "Momentum of RMSProp [0.9]")
flags.DEFINE_float("decay", 0.95, "Decay of RMSProp [0.95]")
flags.DEFINE_float("dropout", 0.9, "Dropout rate")
flags.DEFINE_float("l2_rate", 0.0, "l2 rate")
flags.DEFINE_string("data_dir", "data", "The name of data directory [data]")
flags.DEFINE_string("dataset", "cnn", "The name of dataset [cnn, dailymail]")
flags.DEFINE_string("log_dir", "log", "Directory name to save the log [log]")
flags.DEFINE_string("load_path", None, "The path to old model. [None]")
flags.DEFINE_string("optim", 'RMS', "The optimizer to use [RMS]")
flags.DEFINE_string("attention", "bilinear", "Attention Mechanism")
flags.DEFINE_string("model", "attentive", "model")
flags.DEFINE_string("activation", 'tanh', "The the last activation layer to use before Softmax loss")
flags.DEFINE_bool("bidirect", True, "Whether use bidirection rnn")

FLAGS = flags.FLAGS

def fetch_model(_type):
  if _type == 'attentive':
    from attentive_model import AttentiveReader as Reader
  elif _type == 'stanford':
    from stanford_model import StanfordReader as Reader
  else:
    raise ValueError(_type)

  return Reader


def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.gpu is not None:
    gpu_list  = define_gpu(FLAGS.gpu)
    os.environ['CUDA_VISIBLE_DEVICES']= ",".join(map(str,gpu_list))

  log_dir = "%s/%s"%(FLAGS.log_dir, time.strftime("%m_%d_%H_%M"))
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    with open(log_dir+'/Flags.js','w') as f:
      json.dump(FLAGS.__flags, f, indent=4)
  else:
    print('log_dir exist %s' % log_dir)
    exit(2)


  with tf.Session() as sess:
    reader = fetch_model(FLAGS.model)
    model = reader(batch_size=FLAGS.batch_size, l2_rate=FLAGS.l2_rate,
                                    vocab_size=FLAGS.vocab_size, 
                                    momentum=FLAGS.momentum, decay=FLAGS.decay,
                                    size=FLAGS.hidden_size,
                                    use_optimizer=FLAGS.optim,
                                    activation=FLAGS.activation,
                                    attention=FLAGS.attention,
                                    D=FLAGS.D,
                                    bidirection=FLAGS.bidirect)
    print " [*] Using GPU: ", str(gpu_list)

    model.train(sess, FLAGS.vocab_size, FLAGS.epoch,
                  FLAGS.data_dir, FLAGS.dataset, log_dir, FLAGS.load_path,
                  FLAGS.data_size, FLAGS.eval_every, FLAGS.dropout)

if __name__ == '__main__':
  tf.app.run()
