import tensorflow as tf
import math
from wtfop.wtfop_ops import random_range
a = tf.constant([100],dtype=tf.int32);
b = tf.constant([1,50,99],dtype=tf.int32)
index,hint = random_range(max=a,hint=b,phy_max=20)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
i,h = sess.run([index,hint])
print(i,h)

