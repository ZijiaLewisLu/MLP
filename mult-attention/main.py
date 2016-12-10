#! /usr/bin/python
import os
import sys
import tensorflow as tf
import time
import json
import numpy as np
from utils import pp
from mdu import batchIter
from mdu import _load, restruct_glove_embedding
# from model import ML_Attention


flags = tf.app.flags

flags.DEFINE_integer("gpu", 3, "the number of gpus to use")
flags.DEFINE_integer("data_size", None, "Number of files to train on")
flags.DEFINE_float("eval_every", 100.0, "Eval every step")
flags.DEFINE_float("save_every", 500.0, "Eval every step")
flags.DEFINE_string("log_dir", "log", "Directory name to save the log [log]")
flags.DEFINE_string("data_dir", "data/squad", "Data")
flags.DEFINE_string("load_path", None, "The path to old model.")
flags.DEFINE_boolean("track", False, "whether to track performance")

flags.DEFINE_integer("epoch", 60, "Epoch to train")
flags.DEFINE_integer("vocab_size", 60000, "The size of vocabulary")
flags.DEFINE_integer("batch_size", 32, "The size of batch images")
flags.DEFINE_integer("embed_size", 300, "Embed size")
flags.DEFINE_integer("hidden_size", 256, "Hidden dimension")
flags.DEFINE_integer("atten_layer", 3, "Num of attention layer")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate")
# flags.DEFINE_float("momentum", 0.9, "Momentum of RMSProp [0.9]")
# flags.DEFINE_float("decay", 0.95, "Decay of RMSProp [0.95]")
flags.DEFINE_float("dropout", 0.9, "Dropout rate")
flags.DEFINE_float("l2_rate", 0.0, "l2 regularization rate")
flags.DEFINE_float("clip_norm", 2.5, "l2 regularization rate")
flags.DEFINE_string("optim", 'Adam', "The optimizer to use")
flags.DEFINE_string("atten", 'concat', "Attention Method")
flags.DEFINE_string("model", 'bow', "Model")
flags.DEFINE_string("init", 'ort', "xav, ort, non, ran")
flags.DEFINE_boolean("glove", False, "whether use glove embedding")


FLAGS = flags.FLAGS

# less often changed parameters
sN = 10
sL = 50
qL = 15
stop_id = 2
val_rate = 0.05
glove_dir = './data/glove_wiki'


def orthogonal_initializer(scale=1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    print 'Warning -- You have opted to use the orthogonal_initializer function'
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if len(shape) == 1:
            a = np.random.normal(0.0, 1.0, shape[0])
            return tf.constant(scale * a, dtype=tf.float32)

        flat_shape = (shape[0], np.prod(shape[1:], dtype=np.int32))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        # print('you have initialized one orthogonal matrix.')
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer


def initialize(sess, saver, load_path=None):
    if not load_path:
        sess.run(tf.initialize_all_variables())
    else:
        if os.path.isdir(load_path):
            fname = tf.train.latest_checkpoint(
                os.path.join(load_path, 'ckpts'))
            assert fname is not None
        else:
            fname = load_path
        print "  Load from %s" % fname
        saver.restore(sess, fname)


def prepare_data(path, data_size=None, size=3185, val_rate=0.05):
    train_data = _load(path)
    validate_data = train_data[-size:]
    train_data = train_data[:-size]

    if data_size:
        train_data = train_data[:data_size]
    vsize = max(20, len(train_data) * val_rate)
    vsize = int(min(vsize, len(validate_data)))
    return train_data, validate_data, vsize


def create_logger(track_dir, to_console=True):
    """tracking high accuracy prediction"""
    import logging
    fname = os.path.join(track_dir, 'tracking.log')
    logger = logging.getLogger('tracker')

    hdlr = logging.FileHandler(fname)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    if to_console:
        stdout = logging.StreamHandler()
        formatter = logging.Formatter('  %(levelname)s %(message)s')
        stdout.setFormatter(formatter)

    logger.addHandler(stdout)
    logger.setLevel(logging.DEBUG)

    return logger


def save_track(data, base_name):
    for i, d in enumerate(data):
        fname = "%s_%d" % (base_name, i)
        np.save(fname, d)


def norm(x):
    if not isinstance(x, np.ndarray):
        x = x.values
    return np.sqrt((x**2).sum())


def create_model(FLAGS, sN=sN, sL=sL, qL=qL):
    
    if FLAGS.model == 'bow':
        from model import ML_Attention as Net
    elif FLAGS.model == 'rr':
        from rr_model import ML_Attention as Net
    # elif FLAGS.model == 'share':
        # from model_share import ML_Project as Net
    else:
        raise ValueError(FLAGS.model)

    if FLAGS.init == 'non':
        initializer = None
    elif FLAGS.init == 'xav':
        initializer = tf.contrib.layers.xavier_initializer()
    elif FLAGS.init == 'ran':
        initializer=tf.truncated_normal_initializer(stddev=0.01)
    elif FLAGS.init == 'ort':
        initializer=orthogonal_initializer()

    with tf.variable_scope('model', initializer=initializer,):

        model = Net(FLAGS.batch_size, sN, sL, qL, FLAGS.vocab_size, FLAGS.embed_size, FLAGS.hidden_size,
                    learning_rate=FLAGS.learning_rate,
                    l2_rate=FLAGS.l2_rate,
                    optim=FLAGS.optim,
                    attention=FLAGS.atten,
                    glove=FLAGS.glove,
                    max_norm=FLAGS.clip_norm,
                    attention_layer=FLAGS.atten_layer,
                    )

    return model

# def check_setting(flags):
#     pp.pprint(flags.FLAGS.__flags)
#     go = raw_input('Do you want to go with these setting? ')
#     while go not in ['Yes', 'y', 'Y', 'yes']:
#         if go == 'pp':
#             pp.pprint(flags.FLAGS.__flags)
#         elif go in ['No', 'n', 'N', 'no']:
#             exit(2)
#         else:
#             go = go.split(' ')
#             if len(go) % 2 != 0:
#                 print ' Not understood, try again\n'
#                 continue
            
#             i = 0
#             while i < len(go):
#                 k = go[i]
#                 v = go[i+1]
#                 if k in flags.FLAGS.__flags:
#                     flags.FLAGS.__flags[k] = v
#                 else:
#                     print '%s is a wrong param'

# def input_with_timeout(prompt, timeout=30.0):
#     import thread
#     from threading
#     print prompt
#     timer = threading.Timer(timeout, thread.interrupt_main)
#     astring

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    go = raw_input('Do you want to go with these setting? ')
    if go not in ['Yes', 'y', 'Y', 'yes']:
        exit(2)

    if FLAGS.gpu is not None:
        # gpu_list  = define_gpu(FLAGS.gpu)
        # print('  Using GPU:%s' % gpu_list)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

    sess = tf.Session()
    model = create_model(FLAGS)
    print '  Model Built'

    if FLAGS.glove:
        fname = os.path.join(glove_dir, 'glove.6B.%dd.txt' % FLAGS.embed_size)
        vocab_path = os.path.join(
            FLAGS.data_dir, "vocab_%d.js" % FLAGS.vocab_size)
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        embedding = restruct_glove_embedding(
            fname, vocab, dim=FLAGS.embed_size)
        sess.run(model.emb.assign(embedding))
        print '  Load Embedding Matrix from %s' % fname

    saver = tf.train.Saver(max_to_keep=15)
    initialize(sess, saver, FLAGS.load_path)
    print '  Variable inited'

    if not FLAGS.glove:
        ids_path = os.path.join(
            FLAGS.data_dir, 'ids_not_glove%d_train.txt' % FLAGS.vocab_size)
    else:
        ids_path = os.path.join(
            FLAGS.data_dir, 'ids_glove%d_train.txt' % FLAGS.vocab_size)
    train_data, validate_data, vsize = prepare_data(
        ids_path, data_size=FLAGS.data_size, val_rate=val_rate)
    print '  Data Loaded from %s' % ids_path

    log_dir = "%s/%s" % (FLAGS.log_dir, time.strftime("%m_%d_%H_%M"))
    save_dir = os.path.join(log_dir, 'ckpts')
    if os.path.exists(log_dir):
        print('log_dir exist %s' % log_dir)
        exit(2)
    os.makedirs(save_dir)
    with open(log_dir + '/Flags.js', 'w') as f:
        json.dump(FLAGS.__flags, f, indent=4)
    print '  Writing log to %s' % log_dir

    tracker = create_logger(log_dir, to_console=True)
    if FLAGS.track:
        track_dir = os.path.join(log_dir, 'track')
        os.makedirs(track_dir)

    # counter = 1
    vcounter = 1
    writer = tf.train.SummaryWriter(log_dir, sess.graph)
    start_time = time.time()
    running_acc = 0.0
    running_loss = 0.0
    max_acc = [0, None, None, None]
    min_loss = [np.inf, None, None, None]
    print '  Start Training'
    tracker.info('  So you know I am working:)')
    sys.stdout.flush()

    for epoch_idx in range(FLAGS.epoch):
        np.random.shuffle(train_data)
        titer = batchIter(FLAGS.batch_size, train_data,
                          sN, sL, qL, stop_id=stop_id, add_stop=(not FLAGS.glove))
        tstep = titer.next()

        for batch_idx, P, p_len, Q, q_len, A in titer:

            rslt = sess.run(
                [
                    model.global_step,
                    model.loss,
                    model.accuracy,
                    model.train_op,
                    model.train_summary, 

                    model.score,
                    model.alignment,
                    model.origin_gv,

                    # model.check_op,
                    # model.mask_print,
                    # model.sn_c_print,

                ],
                feed_dict={
                    model.passage: P,
                    model.p_len: p_len,
                    model.query: Q,
                    model.q_len: q_len,
                    model.answer: A,
                    model.dropout: FLAGS.dropout,
                })

            gstep, loss, accuracy, _, sum_str = rslt[:5]
            rslt = rslt[5:]

            score, align, origin_gv = rslt[:3]
            rslt = rslt[3:]

            loss = loss.mean()
            running_acc += accuracy
            running_loss += loss
            writer.add_summary(sum_str, gstep)

            # gradient norm check =============================
            for i, (g, v) in enumerate(origin_gv):
                nm = norm(g)
                if nm > FLAGS.clip_norm:
                    tracker.warning('%s, gradient norm: %f, global_step:%d' % (
                        model.origin_gv[i][1].name, nm, gstep))

            if FLAGS.track:
                if accuracy > max_acc[0]:
                    max_acc = [accuracy, score, align, A]
                if loss < min_loss[0]:
                    min_loss = [loss, score, align, A]

            if (gstep + 1) % 20 == 0:
                print "Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f" \
                    % (epoch_idx, batch_idx, tstep, time.time() - start_time, running_loss / 20.0, running_acc / 20.0)
                sys.stdout.flush()
                running_loss = 0.0
                running_acc = 0.0
                sess.run(model.learning_rate)

            if (gstep + 1) % FLAGS.eval_every == 0:

                if FLAGS.track:
                    mark = str(time.time())
                    base_name = os.path.join(track_dir, mark)

                    tracker.info('Train %d %s' % (gstep, mark))
                    tracker.info('max_accuracy %.4f' % max_acc[0])
                    tracker.info('min_loss %.4f' % min_loss[0])
                    save_track(max_acc[1:], base_name + "_Tacc")
                    save_track(min_loss[1:], base_name + "_Tloss")
                    max_acc = [0, None, None, None]
                    min_loss = [np.inf, None, None, None]

                _accuracy = 0.0
                _loss = 0.0
                idxs = np.random.choice(len(validate_data), size=vsize)
                D = [validate_data[idx] for idx in idxs]
                viter = batchIter(FLAGS.batch_size, D, sN,
                                  sL, qL, stop_id=stop_id, add_stop=(not FLAGS.glove))
                vstep = float(viter.next())
                for batch_idx, P, p_len, Q, q_len, A in viter:
                    loss, accuracy, sum_str = sess.run(
                        [model.loss, model.accuracy, model.validate_summary],
                        feed_dict={
                            model.passage: P,
                            model.p_len: p_len,
                            model.query: Q,
                            model.q_len: q_len,
                            model.answer: A,
                            model.dropout: 1.0,
                        })

                    loss = loss.mean()
                    _accuracy += accuracy
                    _loss += loss
                    vcounter += 1
                    writer.add_summary(sum_str, vcounter)

                    if FLAGS.track:
                        if accuracy > max_acc[0]:
                            max_acc = [accuracy, score, align, A]
                        if loss < min_loss[0]:
                            min_loss = [loss, score, align, A]

                print '  Evaluation: time: %4.4f, loss: %.8f, accuracy: %.8f' % \
                    (time.time() - start_time, _loss / vstep, _accuracy / vstep)
                vcounter += int(vstep / 4.0)  # add gap

                if FLAGS.track:
                    tracker.info('Validate %d %s' % (gstep, mark))
                    tracker.info('max_accuracy %.4f' % max_acc[0])
                    tracker.info('min_loss %.4f' % min_loss[0])
                    save_track(max_acc[1:], base_name + "_Vacc")
                    save_track(min_loss[1:], base_name + "_Vloss")
                    max_acc = [0, None, None, None]
                    min_loss = [np.inf, None, None, None]

            if (gstep + 1) % FLAGS.save_every == 0:
                fname = os.path.join(save_dir, 'model')
                print "  Saving Model..."
                saver.save(sess, fname, global_step=gstep)

if __name__ == '__main__':
    tf.app.run()
    # try:
    #     tf.app.run()
    # except Exception, e:
    #     import ipdb, traceback
    #     etype, value, tb = sys.exc_info()
    #     traceback.print_exc()
    #     ipdb.post_mortem(tb)
