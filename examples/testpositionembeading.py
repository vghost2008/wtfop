import tensorflow as tf
import math
from wtfop.wtfop_ops import position_embedding
size=[3,5]
output = position_embedding(size=size)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
i=1.
p=1.
print(p,i*2,math.sin(p/math.pow(10000.,2*i/5)))
i=1.
p=2.
print(p,i*2+1,math.cos(p/math.pow(10000.,2*i/5)))
print("Result:",sess.run(output))

