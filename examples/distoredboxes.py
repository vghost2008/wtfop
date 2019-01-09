from __future__ import print_function

import tensorflow as tf
import numpy as np
from wtfop.wtfop_ops import distored_boxes
def box_vol(boxes):
	boxes = np.transpose(boxes)
	ymin,xmin,ymax,xmax = boxes[0],boxes[1],boxes[2],boxes[3]
	vol = (ymax-ymin)*(xmax-xmin)
	return vol
sess = tf.Session()
boxes = tf.Variable([[0.0,0.0,0.2,0.2],[0.4,0.4,1.0,1.0]]);
#out_boxes = distored_boxes(boxes,xoffset=[0.5,1.,-0.5,-1.])
out_boxes = distored_boxes(boxes,xoffset=[1.,-1.])

sess.run(tf.global_variables_initializer())

print("gboxes:",sess.run(boxes))
ob = sess.run(out_boxes)
print("boxes",ob)
print("boxes_vol",box_vol(ob))
