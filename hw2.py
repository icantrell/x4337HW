#author: Issiah Cantrell
import numpy as np
import tensorflow as tf

def ws(v):
    with tf.name_scope('summaries'):
        tf.summary.tensor_summary('test_mean',10)

with tf.name_scope('Input_placeholder'):
    a = tf.placeholder(dtype = tf.float32,shape = (10))
    ws(a)

with tf.name_scope('Middle_section'):
    b = tf.reduce_prod(a)
    c = tf.reduce_mean(a)
    d = tf.reduce_sum(a)
    e = tf.add(c,b)

with tf.name_scope('Final_node'):
    f = tf.multiply(e,d)

rv = np.random.normal(size=(10), loc = 1.0, scale = 2.0)
sess = tf.Session()
sess.run(f, feed_dict = {a: rv })
tf.summary.merge_all()
file_writer = tf.summary.FileWriter('../events',sess.graph)
file_writer.close()
sess.close()
