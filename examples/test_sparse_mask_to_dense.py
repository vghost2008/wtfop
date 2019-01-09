from __future__ import print_function
import object_detection.bboxes as odb
import tensorflow as tf
import numpy as np
import semantic.toolkit as smt
from wtfop.wtfop_ops import sparse_mask_to_dense

from wtfop.wtfop_ops import boxes_encode1,decode_boxes1
sess = tf.Session()
a = tf.Variable([5,2,2,2],dtype=tf.int32);
#a = tf.Variable([0,5],dtype=tf.int32);

b = tf.Variable(tf.ones([4,4,2],dtype=tf.bool));
b22 = tf.Variable(tf.ones([2,2],dtype=tf.bool))
b0 = tf.pad(b22,paddings=[[0,2],[0,2]])
b0 = tf.expand_dims(b0,axis=2)
b1 = tf.pad(b22,paddings=[[2,0],[2,0]])
b1 = tf.expand_dims(b1,axis=2)
b2 = tf.pad(b22,paddings=[[2,0],[0,2]])
b2 = tf.expand_dims(b2,axis=2)
b = tf.concat([b0,b1,b1,b2],axis=2)
B = tf.transpose(b,perm=[2,0,1])
c = sparse_mask_to_dense(mask=b,labels=a,num_classes=7,set_background=True)
c = tf.transpose(c,perm=[2,0,1])
sess.run(tf.global_variables_initializer())
print("c:",sess.run(c))
