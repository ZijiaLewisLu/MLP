from GPU_availability import GPU_availability as GPU
import os

def define_gpu(num):
    gpu_list = GPU()[:num]
    gpu_str  = ','.join( map(str, gpu_list) )
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_str 
    return gpu_list

def _construct_parallel(*args, **kwargs):
    """
    Input:
    First three args should be model_builder, batch_size, gpu_list.
    Other args and kwargs will be passed into model_builder.
    Model_builder should be callable, and return a model class with attribute: model.loss, model.accuracy, model.input_dict, model.optim,
    Gpu_list should be a list of usable gpus id
    
    Return:
    """
    raise NotImplemented

def mean(L):
    n = float(len(L))
    m = reduce( lambda x,y: x+y, L) / n
    return m

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
