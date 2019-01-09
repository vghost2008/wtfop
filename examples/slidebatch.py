from __future__ import print_function

import tensorflow as tf
import numpy as np

from wtfop.wtfop_ops import slide_batch
sess = tf.Session()
data = tf.Variable([[0.0,0.0,0.2,0.2],[0.3,0.3,0.5,0.6],[0.1,0.1,0.4,0.4],[0.7,0.7,0.9,0.8]])
#filter=tf.Variable([[0.,0.],[0.,0.]])
filter=tf.Variable([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
out_data = slide_batch(data=data,filter=filter,strides=[2,2],padding="SAME")
print(out_data.get_shape().as_list())
out_data = tf.squeeze(out_data,axis=4)
shape = tf.shape(out_data)
sess.run(tf.global_variables_initializer())
print("data:",sess.run(data))
print("filter",sess.run(filter))
print("out_data:",sess.run(out_data))
print("out_shape:",sess.run(shape))
