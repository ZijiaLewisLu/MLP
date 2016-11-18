from GPU_availability import GPU_availability as GPU
import os
import tensorflow as tf
import numpy as np


def define_gpu(num):
    gpu_list = GPU()[:num]
    gpu_str = ','.join(map(str, gpu_list))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    return gpu_list


class MultiGPU_Manager(object):

    def __init__(self, gpu_list, model_builder, session=None):
        """ 
        Inputs:
            gpu_list: the id of gpus to use
            model_builder: a func or class that builds the graph.
                           it should requires no arguments and return a model object.
                           three attributes model objects must have:
                                model.loss, model.accuracy, model.optim (optimizer)
            session: default session. If None, new one is created.
                     If session is created externally, make sure to set allow_soft_placement=True.
                     Otherwise error may be encountered when loading checkpoints.
        """
        self.N = len(gpu_list)
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
        with tf.device('/gpu:%d' % self.main_gpu):
            self.main_model = self.model_builder()
        self.variables = tf.all_variables()
        self.variable_dict = {v.name: v for v in self.variables}

        # create other models
        models = [self.main_model]
        for g in self.gpu_list[1:]:
            with tf.device('/gpu:%d' % g), tf.variable_scope("GPU%d" % g):
                models.append(self.model_builder())
        self.GM_dict = dict(zip(self.gpu_list, models))
        self.models = models

        # create total loss and accuracy
        self.loss = self._mean([m.loss for m in models])
        self.accuracy = self._mean([m.accuracy for m in models])

        # create update_op
        gvs = []
        for m in models:
            gvs += m.optim.compute_gradients(m.loss)
        gv_dict = {_.name: [] for _ in self.variables}
        for g, v in gvs:
            if g is not None:
                if v.name.startswith('GPU'):
                    k = '/'.join(v.name.split('/')[1:])
                else:
                    k = v.name
                gv_dict[k].append(g)
        the_gv = [(self._mean(gv_dict[v.name]), v) for v in self.variables]
        self.gv_dict = gv_dict
        self.update_op = self.main_model.optim.apply_gradients(the_gv)

        # create assign_op to sync values
        assign_op_dict = {_.name: [_] for _ in self.variables}
        for v in tf.all_variables():
            if v.name.startswith('GPU'):
                k = '/'.join(v.name.split('/')[1:])
                main_v = assign_op_dict[k][0]
                assign_op_dict[k].append(v.assign(main_v.value()))
        assign_op_dict = {k: tf.group(*v[1:])
                          for k, v in assign_op_dict.items()}
        self.assign_op_dict = assign_op_dict
        self.assign_op = tf.group(*assign_op_dict.values())

        # create train_op
        # apply gradient to main model and sync across gpus
        with tf.control_dependencies([self.update_op]):
            self.train_op = tf.group(*assign_op_dict.values())
        self.t_op = tf.group(self.assign_op, self.update_op)

        # build saver
        self.saver = tf.train.Saver(self.variables)

    def _mean(self, L):
        n = float(len(L))
        m = reduce(lambda x, y: x + y, L) / n
        return m

    def _load(self, load_path):
        assert load_path is not None, 'No Checkpoint specified.'
        assert self.saver is not None, 'No Saver'
        if os.path.isdir(self.load_path):
            ckpt_name = tf.train.latest_checkpoint(self.load_path)
        else:
            ckpt_name = self.load_path
        self.saver.restore(self.sess, ckpt_name)
        self.sess.run(self.assign_op)

    def init_variable(self, load_path=None):
        """
        Initialize all variables of the graph.
        Keyword Args - load_path: 
            If None, run tf.initialize_all_variables.
            If load_path is a checkpoint file, directly restore parameter from files.
            If it's a directory, load the latest checkpoint file. 
        """
        if load_path is None:
            init = tf.initialize_all_variables()
            self.sess.run(init)
            self.sess.run(self.assign_op)
        else:
            self.load_path = load_path
            self._load(load_path)

    def feed_dict(self, feed_dict, batch_dim=0):
        """
        Split input data and assign them to each model inputs accordingly.
        Inputs:
            feed_dict: a {str: data} dictionary.
                       the keyword strings should be the names of model attributes.
            batch_dim: the dim of data to split on, 0 by default.

        Outputs:
            a new { model.attr: data } dict.
        """
        fd = {}
        for k, v in feed_dict.items():
            vs = np.split(v, self.N, axis=batch_dim)
            fd.update({getattr(m, k): d for m, d in zip(self.models, vs)})
        return fd

    def save(self, save_path, **kwargs):
        """
        Save checkpoint
        Inputs:
            save_path: file name to save in
            kwargs: other kwargs to pass to Saver.save()
        """
        assert self.saver is not None
        self.saver.save(self.sess, save_path, **kwargs)

    def get_list(self, arg):
        """
        A thin wrapper to help collect certain attirbutes from all models:
        Inputs:
            arg: string or callable
                 If string, collect model attributes specified by the string
                 If callable, pass each model into arg and collect outputs

        Outputs:
            a list of required arg.
        """
        if isinstance(arg, str):
            return [getattr(m, arg) for m in self.models]
        elif callable(arg):
            return map(arg, self.models)
        else:
            raise ValueError('Unknown Arg Type')