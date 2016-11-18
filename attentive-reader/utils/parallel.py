from GPU_availability import GPU_availability as GPU
import os
import tensorflow as tf
import numpy as np

def define_gpu(num):
    gpu_list = GPU()[:num]
    gpu_str  = ','.join( map(str, gpu_list) )
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_str 
    return gpu_list

class MultiGPU_Manager(object):

    def __init__(self, gpu_list, model_builder, session=None):
        """ session should have: config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=True)
        """
        self.N = N = len(gpu_list)
        self.gpu_list = gpu_list
        self.model_builder = model_builder
        assert callable(model_builder), 'Model_builder is not callable'

        self.saver = None
        if session is None:
            self.sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True))
        else:
            self.sess = session

        self.main_gpu = None
        self.main_model = None
        self.models = None
        self.GM_dict = None

        self.loss = None
        self.accuracy = None
        self.train_op = None

        self._build()

    def _build(self):
        # create main model
        self.main_gpu = self.gpu_list[0]
        with tf.device('/gpu:%d'%self.main_gpu):
            self.main_model = self.model_builder()

        self.variables = tf.all_variables() 

        # create other models
        models = [self.main_model]
        for g in self.gpu_list[1:]:
            with tf.device('/gpu:%d'%g), tf.variable_scope("GPU%d"%g):
                models.append(self.model_builder())
        self.GM_dict = dict( zip(self.gpu_list, models) )
        self.models = models

        # create assign_op to sync values
        assign_op_dict = { _.name:[_] for _ in self.variables }
        for v in tf.all_variables():
            if v.name.startswith('GPU'):
                k = '/'.join( v.name.split('/')[1:] )
                main_v = assign_op_dict[k][0]
                assign_op_dict[k].append( v.assign(main_v.value()) )
        assign_op_dict = { k: tf.group(*v[1:]) for k,v in assign_op_dict.items() }
        self.assign_op_dict = assign_op_dict
        self.assign_op = tf.group( *assign_op_dict.values() )

        # create total loss and accuracy
        self.loss = self._mean([ m.loss for m in models ])
        self.accuracy = self._mean([ m.accuracy for m in models ])

        # create train_op!
        gvs = []
        for m in models:
            gvs += m.optim.compute_gradients(m.loss)
        gv_dict = { _.name:[] for _ in self.variables }
        for g, v in gvs:
            if g is not None:
                if v.name.startswith('GPU'):
                    k = '/'.join( v.name.split('/')[1:] )
                else:
                    k = v.name
                gv_dict[k].append(g)

        the_gv = [ ( self._mean(gv_dict[v.name]), v )  for v in self.variables  ]
        self.update_op = self.main_model.optim.apply_gradients( the_gv )
        ## apply gradient to main model and sync across gpus
        with tf.control_dependencies([self.update_op]):
            self.train_op = tf.group(*assign_op_dict.values())
        # self.train_op = tf.group( self.assign_op, self.update_op )  
        self.gv_dict = gv_dict

        # build saver
        self.saver = tf.train.Saver(self.variables)

    def _mean(self, L):
        n = float(len(L))
        m = reduce( lambda x,y: x+y, L) / n
        return m

    def _load(self, load_path):
        assert load_path is not None, 'No Checkpoint specified.'
        assert self.saver is not None
        if os.path.isdir(self.load_path):
            ckpt_name = tf.train.latest_checkpoint(self.load_path)
        else:
            ckpt_name = self.load_path
        self.saver.restore(self.sess, ckpt_name)
        self.sess.run( self.assign_op )


    def init_variable(self, load_path=None):
        if load_path is None:
            init = tf.initialize_all_variables()
            self.sess.run( init )
            self.sess.run( self.assign_op )
        else:
            self.load_path = load_path
            self._load(load_path)


    def feed_dict(self, feed_dict, batch_dim=0):
        """batch dim are set to dim 0 as default"""
        fd = {}
        for k,v in feed_dict.items():
            vs = np.split( v, self.N, axis=batch_dim )
            fd.update({ getattr(m,k):d for m,d in zip(self.models, vs) })
        return fd

    def save(self, save_path, **kwargs):
        assert self.saver is not None
        # if os.path.exists(save_path):
            # os.makedirs(save_path)
        self.saver.save(self.sess, save_path, **kwargs)



def parallel(models, real_batch_size, gpu_list, learning_rate):
    """
    real_batch_size is the batch_size single model expects
    """
    import tensorflow as tf
    assert len(models) == len(gpu_list)
    N = len(gpu_list)

    # split input
    BS = real_batch_size*N
    the_doc = tf.placeholder(tf.int32, [BS, models[0].max_nsteps])
    the_query = tf.placeholder(tf.int32, [BS, models[0].max_query_length])
    the_d_end = tf.placeholder(tf.int32, BS)
    the_q_end = tf.placeholder(tf.int32, BS)
    the_y = tf.placeholder(tf.float32, [BS, models[0].vocab_size])

    the_doc = tf.split(0,N,the_doc)
    the_query = tf.split(0,N,the_query)
    the_d_end = tf.split(0,N,the_d_end)
    the_q_end = tf.split(0,N,the_q_end)
    the_y  = tf.split(0,N,the_y)
    for i in range(N):
        models[i].document = the_doc[i]
        models[i].query = the_query[i]
        models[i].d_end = the_d_end[i]
        models[i].q_end = the_q_end[i]
        models[i].y = the_y[i]

    input_list = [the_doc, the_query, the_d_end, the_q_end, the_y]


    # create model
    gm_pair = zip(gpu_list, models)
    for g,m in gm_pair:
        with tf.device('/gpu:%d'% g), tf.variable_scope("GPU%d"%g):
            m.prepare_model(parallel=True)

    ## sync gradient      
    for i in range(len(models[0].grad_and_var)):
        g = mean([ m.grad_and_var[i][0] for m in models ])
        for m in models:
            print(m.grad_and_var[i][1].name) 
            m.grad_and_var[i] = (g, m.grad_and_var[i][1]) 
        print()

    for m in models:
        m.train_op = m.optim.apply_gradients(m.grad_and_var)

    gm_dict = dict(gm_pair)
    the_loss = mean([ m.loss for m in models ])
    the_acc  = mean([ m.accuracy for m in models ])
    the_train_op = tf.group( [m.train_op for m in models] )
    return the_train_op, the_loss, the_acc, input_list, gm_dict
