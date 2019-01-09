from __future__ import print_function

import tensorflow as tf
import numpy as np

from wtfop.wtfop_ops import draw_points
sess = tf.Session()
a = tf.ones([3,4,5])
points = tf.constant([[2,1]],dtype=tf.float32)
b = draw_points(image=a,points=points,color=[0.],point_size=0)
sess.run(tf.global_variables_initializer())
print(sess.run(b))

print("AAA")
print(sess.run(a))

