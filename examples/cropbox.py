from __future__ import print_function

import tensorflow as tf
import object_detection.npod_toolkit as npod
import numpy as np

from wtfop.wtfop_ops import crop_boxes

np_gboxes = np.array([[0.0,0.0,0.2,0.2],[0.3,0.3,0.5,0.6],[0.1,0.1,0.4,0.4],[0.7,0.7,0.9,0.8]])
np_subbox = np.array([0.1,0.1,0.7,0.7])
print(npod.crop_box(bboxes=np_gboxes,sub_box=np_subbox,remove_threshold=0.8))
sess = tf.Session()
gboxes = tf.Variable(np_gboxes,dtype=tf.float32);
labels = tf.Variable([1,2,3,4])
ref_boxes = tf.Variable(np_subbox,dtype=tf.float32);
boxes,mask = crop_boxes(ref_box=ref_boxes,boxes=gboxes,threshold=0.8)
labels = tf.boolean_mask(labels,mask)
sess.run(tf.global_variables_initializer())
print("gboxes:",sess.run(boxes))
print("mask",sess.run(mask))
print("boxes",sess.run(labels))
