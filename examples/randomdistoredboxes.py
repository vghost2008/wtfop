from __future__ import print_function

import tensorflow as tf
import numpy as np
from wtfop.wtfop_ops import random_distored_boxes
def box_vol(boxes):
	boxes = np.transpose(boxes)
	ymin,xmin,ymax,xmax = boxes[0],boxes[1],boxes[2],boxes[3]
	vol = (ymax-ymin)*(xmax-xmin)
	return vol
sess = tf.Session()
boxes = tf.Variable([[0.0,0.0,0.2,0.2],[0.4,0.4,1.0,1.0]]);
out_boxes = random_distored_boxes(boxes,limits=[0.2,0.,0.2],size=1,keep_org=True)

sess.run(tf.global_variables_initializer())

print("gboxes:",sess.run(boxes))
ob = sess.run(out_boxes)
print("boxes",ob)
print("boxes_vol",box_vol(ob))
