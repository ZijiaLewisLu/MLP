import tensorflow as tf
import numpy as np
import os

class Mlp():
    def __init__(self, parallel=True):
        x = tf.placeholder(tf.float32, [3,10])
        y = tf.placeholder(tf.int32,  [3,2])

        w = tf.get_variable('w', shape=[10,2], dtype=tf.float32)
        score = tf.matmul(x,w)

        loss = tf.nn.softmax_cross_entropy_with_logits(score,y)
        prd = tf.argmax(score,1)
        y_  = tf.argmax(y,1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prd,y_),tf.float32))

        optim = tf.train.GradientDescentOptimizer(0.05)
        self.x = x
        self.y = y
        self.loss = loss
        self.accuracy = accuracy
        self.optim = optim
        if not parallel:
            self.train_op = optim.minimize(loss)

def train():
    X = np.random.rand(3,10)
    Y = np.zeros([3,2])
    Y[[0,1,2],[1,1,0]] = 1
    
    m = Mlp(False)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run( tf.initialize_all_variables() )
        for i in range(40):
            _, a, l = sess.run( [ m.train_op, m.accuracy, m.loss ], feed_dict={ m.x:X, m.y:Y})
            print i, a, l
    
        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        saver.save(sess, "tmp/ckpt")
    

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    train()

