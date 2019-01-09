from __future__ import print_function

import tensorflow as tf
import numpy as np

from wtfop.wtfop_ops import probability_adjust
sess = tf.Session()
probs = tf.Variable([[0.0,0.0,0.2,0.2],[0.3,0.3,0.5,0.6],[0.1,0.1,0.4,0.4],[0.7,0.7,0.9,0.8]]);
out_probs  = probability_adjust(probs=probs,classes=[])
sess.run(tf.global_variables_initializer())
print("probs:",sess.run(probs))
print("out_probs:",sess.run(out_probs))
