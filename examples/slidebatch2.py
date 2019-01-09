from __future__ import print_function

import tensorflow as tf
import numpy as np

from wtfop.wtfop_ops import slide_batch
sess = tf.Session()
src = tf.range(460*600*3,dtype=tf.float32)
data = tf.reshape(src,[460,600,3])
#filter=tf.Variable([[0.,0.],[0.,0.]])
filter=tf.zeros([21,21,3])
out_data = slide_batch(data=data,filter=filter,strides=[10,10],padding="SAME")
print(out_data.get_shape().as_list())
#out_data = tf.squeeze(out_data,axis=4)
shape = tf.shape(out_data)
sess.run(tf.global_variables_initializer())
print("data:",sess.run(data))
print("filter",sess.run(filter))
print("out_data:",sess.run(out_data))
print("out_shape:",sess.run(shape))
