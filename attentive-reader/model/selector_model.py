import tensorflow as tf
# from tensorflow.python.ops import rnn_cell
from base import BaseModel
import numpy as np
import h5py

import time

def fetch_data(fname='./FULL_BIG_BOOST.h5', val_size=3000):
    with h5py.File(fname, 'r') as hf:
        doc = hf['doc'][()]
        dlen = hf['dlen'][()]
        que = hf['que'][()]
        qlen = hf['qlen'][()]
        y = hf['label'][()]
    
    data = [ doc, dlen, que, qlen, y ]
    train_data = [ _[val_size:] for _ in data ]
    validate_data = [ _[:val_size] for _ in data ]
    return train_data, validate_data

class Selector(BaseModel):
    """Attentive Reader."""

    def prepare_model(self, parallel=False):

        self.construct_inputs()
        # self.attention = 'concat'

        # Embeding
        self.emb = tf.get_variable("emb", [self.vocab_size, self.size])
        embed_d = tf.nn.embedding_lookup(self.emb, self.document, name='embed_d')
        embed_q = tf.nn.embedding_lookup(self.emb, self.query, name='embed_q')

        embed_d = tf.nn.dropout(embed_d, keep_prob=self.dropout)
        embed_q = tf.nn.dropout(embed_q, keep_prob=self.dropout)

        # representation
        with tf.variable_scope("document_represent"):
            # d_t: N, T, Hidden
            d_t, d_final_state = self.rnn( self.size, embed_d, self.d_end, use_bidirection=self.bidirection)
            d_t = tf.concat(2, d_t)

        with tf.variable_scope("query_represent"):
            q_t, q_final_state = self.rnn( self.size, embed_q, self.q_end, use_bidirection=self.bidirection)
            u = self.extract_rnn_state( self.bidirection, q_t, self.q_end )


        d_t = tf.nn.dropout(d_t, keep_prob=self.dropout)
        u = tf.nn.dropout(u, keep_prob=self.dropout)
        self.d_t = d_t
        self.u = u

        # attention
        r = self.apply_attention(self.attention, 2*self.size, d_t, u, 'concat')

        # predict
        W_rg = tf.get_variable("W_rg", [2 * self.size, self.size])
        W_ug = tf.get_variable("W_ug", [2 * self.size, self.size])
        W_g = tf.get_variable('W_g', [self.size, 1])
        mid = tf.matmul(r, W_rg, name='r_x_W') + \
            tf.matmul(u, W_ug, name='u_x_W')
        g = tf.tanh(mid, name='g')
        g = tf.matmul(g, W_g, name='g_x_W')
        self.score = g

        # beact_sum = tf.scalar_summary(
        #     'before activitation', tf.reduce_mean(mid))
        # afact_sum = tf.scalar_summary(
        #     'before activitation_after', tf.reduce_mean(g))

        self.construct_loss_and_summary(self.score)

    def construct_inputs(self, label_dim=1):
        self.document = tf.placeholder(
            tf.int32, [self.batch_size, self.max_nsteps], name='document')
        self.query = tf.placeholder(
            tf.int32, [self.batch_size, self.max_query_length], name='query')
        self.d_end = tf.placeholder(tf.int32, self.batch_size, name='docu-end')
        self.q_end = tf.placeholder(tf.int32, self.batch_size, name='quer-end')
        self.label = tf.placeholder(
            tf.float32, [self.batch_size, 1], name='Y')
        self.dropout = tf.placeholder(tf.float32, name='dropout_rate')
        
    def construct_loss_and_summary(self, score, parallel=False):

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
            score, self.label, name='loss')
        loss_sum = tf.scalar_summary("T_loss", tf.reduce_mean(self.loss))
        
        self.prediction = tf.greater(score, 0, name='prediction')
        pred = tf.to_float(self.prediction)

        self.correct = tf.equal(
            pred, self.label, name='correct_prediction')
        self.accuracy = tf.reduce_mean(
            tf.to_float(self.correct), name='accuracy')
        acc_sum = tf.scalar_summary("T_accuracy", self.accuracy)

        # optimize
        self.optim = self.get_optimizer()

        self.grad_and_var = self.optim.compute_gradients(self.loss)
        with tf.name_scope('clip_norm'):
            new = []
            for _g, v in self.grad_and_var:
                if _g is not None:
                    new.append( (tf.clip_by_norm(_g, self.max_norm), v) )
                else:
                    new.append( (_g,v) )

            self.grad_and_var = new                            

        if not parallel:
            self.train_op = self.optim.apply_gradients(
                self.grad_and_var, name='train_op')
        else:
            self.train_op = None

        # train_sum
        gv_sum = []
        zf = []
        for g, v in self.grad_and_var:
            v_sum = tf.scalar_summary(
                "I_{}-var/mean".format(v.name), tf.reduce_mean(v))
            gv_sum.append(v_sum)
            if g is not None:
                g_sum = tf.scalar_summary(
                    "I_{}-grad/mean".format(v.name), tf.reduce_mean(g))
                zero_frac = tf.scalar_summary(
                    "I_{}-grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                gv_sum.append(g_sum)
                zf.append(zero_frac)

        self.vname = [v.name for g, v in self.grad_and_var]
        self.vars = [v for g, v in self.grad_and_var]
        self.gras = [g for g, v in self.grad_and_var if g is not None]
        self.gname = [ v.name for g, v in self.grad_and_var if g is not None ]

        if self.attention == 'local':
            self.train_sum = tf.merge_summary([loss_sum, acc_sum])
        else:
            self.train_sum = tf.merge_summary([loss_sum, acc_sum])

        # validation sum
        v_loss_sum = tf.scalar_summary("V_loss", tf.reduce_mean(self.loss))
        v_acc_sum = tf.scalar_summary("V_accuracy", self.accuracy)

        embed_sum = tf.histogram_summary("embed", self.emb)
        self.validate_sum = tf.merge_summary(
            [embed_sum, v_loss_sum, v_acc_sum])
   
    def step(self, sess, data, fetch, dropout_rate): 
        batch_idx, docs, d_end, queries, q_end, label = data
        
        # not the common word
        mask = (label[:,1] == 0)
        # print mask.mean()
        label = (label[:,1] == 0).astype(np.int)
        label[mask] = 0.2
        label[np.invert(mask)] = 0.8
        label = np.expand_dims(label, -1)
        
        rslt = sess.run( fetch,
                feed_dict={ self.document: docs,
                            self.query: queries,
                            self.d_end: d_end,
                            self.q_end: q_end,
                            self.label: label,
                            self.dropout: dropout_rate,
                             }
                           )
        return rslt
       
    def train(self, sess, vocab_size, epoch=25, data_dir="data", dataset_name="cnn",
              log_dir='log/tmp/', load_path=None, data_size=3000, eval_every=1500, val_rate=0.1, dropout_rate=0.9):

        print(" [*] Building Network...")
        start = time.time()
        self.prepare_model()
        print(" [*] Preparing model finished. Use %4.4f" %
              (time.time() - start))

        # Summary
        writer = tf.train.SummaryWriter(log_dir, sess.graph)
        print(" [*] Writing log to %s" % log_dir)

        # Saver and Load
        self.saver = tf.train.Saver(max_to_keep=15)
        if load_path is not None:
            if os.path.isdir(load_path):
                fname = tf.train.latest_checkpoint(
                    os.path.join(load_path, 'ckpts'))
                assert fname is not None
            else:
                fname = load_path

            print(" [*] Loading %s" % fname)
            self.saver.restore(sess, fname)
            print(" [*] Checkpoint is loaded.")
        else:
            sess.run(tf.initialize_all_variables())
            print(" [*] No checkpoint to load, all variable inited")

        counter = 0
        vcounter = 0
        start_time = time.time()
        ACC = []
        LOSS = []
        train_data, validate_data = fetch_data()
        
        if data_size:
            train_data = [ _[:data_size] for _ in train_data ]
        validate_size = int(
            min(max(20.0, float(len(train_data[0])) * val_rate), len(validate_data[0])))
        print(" [*] Validate_size %d" % validate_size)

        for epoch_idx in xrange(epoch):
            # load data
            train_iter = data_iter(self.batch_size, train_data, shuffle_data=True)
            tsteps = train_iter.next()

            # train
            running_acc = 0
            running_loss = 0
            for data in train_iter:
                batch_idx, docs, d_end, queries, q_end, y = data
                _, summary_str, cost, accuracy, gs, pred = self.step( sess, data, 
                                [self.train_op, self.train_sum, self.loss, self.accuracy,
                                    self.gras,
                                    self.prediction
                                ], dropout_rate)

                writer.add_summary(summary_str, counter)
                running_acc += accuracy
                running_loss += np.mean(cost)
                if counter % 10 == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, accuracy: %.8f"
                          % (epoch_idx, batch_idx, tsteps, time.time() - start_time, running_loss / 10.0, running_acc / 10.0))
                    running_loss = 0
                    running_acc = 0
                counter += 1

                if (counter+1) % 30 == 0:
                    print '-----------------check prediction----------------'
                    print (y[:,1]==0).mean(), pred.mean()
                
                if False:
                    for name, g in zip(self.gname, gs):
                        _n = norm(g)
                        if _n > self.max_norm:
                            print 'EXPLODE %s %f' % ( name, _n )

                if (counter + 1) % eval_every == 0:
                    # validate
                    vrunning_acc = 0
                    vrunning_loss = 0

                    idxs = np.random.choice(
                        len(validate_data), size=validate_size)
                    _vdata = [ _[idxs] for _ in validate_data ]
                    validate_iter = data_iter(self.batch_size, _vdata, shuffle_data=True)
                    vsteps = validate_iter.next()

                    for data in validate_iter:
                        batch_idx, docs, d_end, queries, q_end, y = data
                        validate_sum_str, cost, accuracy, pred = self.step( sess, data, 
                                                                     [self.validate_sum, self.loss, self.accuracy, self.prediction],
                                                                    1.0)
                        writer.add_summary(validate_sum_str, vcounter)
                        vrunning_acc += accuracy
                        vrunning_loss += np.mean(cost)
                        vcounter += 1 
                        
                        if batch_idx % 20 == 0:
                            print '-----------------[V]check prediction----------------'
                            print (y[:,1]==0).mean(), pred.mean()

                    ACC.append(vrunning_acc / vsteps)
                    LOSS.append(vrunning_loss / vsteps)
                    vcounter += vsteps
                    print("Epoch: [%2d] Validation time: %4.4f, loss: %.8f, accuracy: %.8f"
                          % (epoch_idx, time.time() - start_time, vrunning_loss / vsteps, vrunning_acc / vsteps))

                    # save
                    self.save(sess, log_dir, global_step=counter)

            print('\n\n')

            
            
def data_iter(batch_size, data, shuffle_data=True):
    doc, d_len, que, q_len, label = data
    N = doc.shape[0]

    steps = np.ceil(N / float(batch_size))
    steps = int(steps)
    yield steps

    oh_label = np.zeros([batch_size, 3], dtype=np.int)
    
    for s in range(steps):

        head = s * batch_size
        end = (s + 1) * batch_size
        if end <= N:
            d = doc[head:end]
            dl = d_len[head:end]
            q = que[head:end]
            ql = q_len[head:end]
            l = label[head:end]
        else:
            d = np.concatenate([doc[head:], doc[:end - N]], axis=0)
            dl = np.concatenate([d_len[head:], d_len[:end - N]], axis=0)
            q = np.concatenate([que[head:], que[:end - N]], axis=0)
            ql = np.concatenate([q_len[head:], q_len[:end - N]], axis=0)
            l = np.concatenate([label[head:], label[:end - N]], axis=0)

        if shuffle_data:
            order = range(d.shape[0])
            np.random.shuffle(order)
            d = d[order]
            dl = dl[order]
            q = q[order]
            ql = ql[order]
            l = l[order]

        oh_label.fill(0)

        try:
            oh_label[range(batch_size), l - 1] = 1
        except:
            import ipdb
            ipdb.set_trace()

        yield s, d, dl, q, ql, oh_label            
