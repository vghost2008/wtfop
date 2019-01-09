from __future__ import print_function

import tensorflow as tf
import numpy as np

from wtfop.wtfop_ops import boxes_encode1,decode_boxes1
sess = tf.Session()
gboxes = tf.Variable([[0.0,0.0,0.1,0.2],[0.3,0.3,0.5,0.6],[0.1,0.1,0.4,0.4],[0.7,0.7,0.9,0.8]]);
glabels = tf.Variable([1,2,3,4]);
boxes = tf.Variable([[0.02,0.02,0.2,0.3],[0.27,0.27,0.51,0.61],[0.101,0.1,0.44,0.4],[0.73,0.71,0.91,0.81]]);
out_boxes,out_labels,out_scores,out_remove_indices = boxes_encode1(boxes,gboxes,glabels,0.5,0.3)
dboxes = decode_boxes1(boxes,out_boxes)
sess.run(tf.global_variables_initializer())
print("gboxes:",sess.run(gboxes))
print("boxes",sess.run(boxes))
print("out_boxes:",sess.run(out_boxes))
print("out_labels",sess.run(out_labels))
print("out_scoures",sess.run(out_scores))
print("out_remove_indices",sess.run(out_remove_indices))
print("dboxes",sess.run(dboxes))
print("delta",np.sum(np.abs(sess.run(dboxes)-sess.run(gboxes))))
