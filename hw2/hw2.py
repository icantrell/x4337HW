#author: Issiah Cantrell
#import libraries
import numpy as np
import tensorflow as tf
#set up placeholder and capture summary
with tf.name_scope('Input_placeholder'):
    a = tf.placeholder(dtype = tf.float32)
    s = tf.summary.tensor_summary('summary',a)

#do tensorflow operations on the data
with tf.name_scope('Middle_section'):
    #reduces vector to scalars
    b = tf.reduce_prod(a)
    c = tf.reduce_mean(a)
    d = tf.reduce_sum(a)

    e = tf.add(c,b)

#multiply last two operations together
with tf.name_scope('Final_node'):
    f = tf.multiply(e,d)

#init random vector
rv = np.random.normal(size=(100), loc = 1.0, scale = 2.0)
#merge all summaries 
merged = tf.summary.merge_all()
#init session
sess = tf.Session()
#init summary writer
file_writer = tf.summary.FileWriter('../events',graph = tf.summary.default_graph())
#run session and feed in random vector to placeholder node
summary, res = sess.run([merged,f], feed_dict = {a: rv })
#add the summaries to summary writer
file_writer.add_summary(summary,0)

#clean up
file_writer.close()
sess.close()
