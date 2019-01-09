from __future__ import print_function

import tensorflow as tf
import numpy as np

from wtfop.wtfop_ops import distored_qa,expand_tensor
sess = tf.Session()
aa = tf.Variable(tf.random_normal([24])*10)
a = tf.Variable(tf.random_normal([6,10])*100)
b = tf.Variable(tf.random_normal([6,12])*10)
a = tf.cast(a,tf.int64)
b = tf.cast(b,tf.int64)
e,f,g = distored_qa(a,b,expand_nr=3)
h = g*3
h = h+1
bb = expand_tensor(a,2)
sess.run(tf.global_variables_initializer())
print("A:",sess.run(a))
print("B:",sess.run(b))

print(sess.run(e))
print(sess.run(f))
print(sess.run(g))
print(sess.run(h))
print(sess.run(aa))
print(sess.run(aa*tf.cast(h,tf.float32)))

print(sess.run(a))
print(sess.run(bb))


