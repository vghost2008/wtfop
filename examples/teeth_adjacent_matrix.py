from __future__ import print_function

import tensorflow as tf
import numpy as np

from wtfop.wtfop_ops import teeth_adjacent_matrix
sess = tf.Session()
bboxes = []
labels = []
'''for i in range(4):
    row = i*0.1
    for j in range(3):
        col = j*0.1
        box = [row-0.01,col-0.01,row+0.01,col+0.01]
        bboxes.append(box)
        labels.append(1)'''
for i in range(2):
    row = i*0.1
    for j in range(2):
        col = j*0.1
        box = [row-0.01,col-0.01,row+0.01,col+0.01]
        bboxes.append(box)
        labels.append(1)
for i in range(2):
    row = i*0.1+10.
    for j in range(2):
        col = j*0.1+10.
        box = [row-0.01,col-0.01,row+0.01,col+0.01]
        bboxes.append(box)
        labels.append(1)
bboxes = np.array(bboxes)
labels = np.array(labels)
i_bboxes = tf.Variable(bboxes,dtype=tf.float32)
i_labels = tf.Variable(labels)
matrix = teeth_adjacent_matrix(boxes=i_bboxes,labels=i_labels,min_nr=4,min_dis=0.01)
#filter=tf.Variable([[0.,0.],[0.,0.]])

#out_data = tf.squeeze(out_data,axis=4)
sess.run(tf.global_variables_initializer())
print("bboxes:",sess.run(i_bboxes))
print("labels",sess.run(i_labels))
print("matrix:",sess.run(matrix))
'''
min_dis=0.15,min_nr=2
bboxes: [[-0.01 -0.01  0.01  0.01]
 [-0.01  0.09  0.01  0.11]
 [-0.01  0.19  0.01  0.21]
 [ 0.09 -0.01  0.11  0.01]
 [ 0.09  0.09  0.11  0.11]
 [ 0.09  0.19  0.11  0.21]
 [ 0.19 -0.01  0.21  0.01]
 [ 0.19  0.09  0.21  0.11]
 [ 0.19  0.19  0.21  0.21]
 [ 0.29 -0.01  0.31  0.01]
 [ 0.29  0.09  0.31  0.11]
 [ 0.29  0.19  0.31  0.21]]
labels [1 1 1 1 1 1 1 1 1 1 1 1]
matrix: [[2 1 0 1 1 0 0 0 0 0 0 0]
 [1 2 1 1 1 1 0 0 0 0 0 0]
 [0 1 2 0 1 1 0 0 0 0 0 0]
 [1 1 0 2 1 0 1 1 0 0 0 0]
 [1 1 1 1 2 1 1 1 1 0 0 0]
 [0 1 1 0 1 2 0 1 1 0 0 0]
 [0 0 0 1 1 0 2 1 0 1 1 0]
 [0 0 0 1 1 1 1 2 1 1 1 1]
 [0 0 0 0 1 1 0 1 2 0 1 1]
 [0 0 0 0 0 0 1 1 0 2 1 0]
 [0 0 0 0 0 0 1 1 1 1 2 1]
 [0 0 0 0 0 0 0 1 1 0 1 2]]
'''
'''
min_nr=3, min_dis=0.01
bboxes: [[-0.01 -0.01  0.01  0.01]
 [-0.01  0.09  0.01  0.11]
 [-0.01  0.19  0.01  0.21]
 [ 0.09 -0.01  0.11  0.01]
 [ 0.09  0.09  0.11  0.11]
 [ 0.09  0.19  0.11  0.21]
 [ 0.19 -0.01  0.21  0.01]
 [ 0.19  0.09  0.21  0.11]
 [ 0.19  0.19  0.21  0.21]
 [ 0.29 -0.01  0.31  0.01]
 [ 0.29  0.09  0.31  0.11]
 [ 0.29  0.19  0.31  0.21]]
labels [1 1 1 1 1 1 1 1 1 1 1 1]
matrix: [[2 3 0 3 0 0 0 0 0 0 0 0]
 [3 2 3 0 0 0 0 0 0 0 0 0]
 [0 3 2 0 0 3 0 0 0 0 0 0]
 [3 0 0 2 0 0 3 0 0 0 0 0]
 [0 0 0 0 2 3 0 3 0 0 0 0]
 [0 0 0 0 3 2 0 0 3 0 0 0]
 [0 0 0 3 0 0 2 0 0 3 0 0]
 [0 0 0 0 3 0 0 2 3 0 0 0]
 [0 0 0 0 0 3 0 3 2 0 0 0]
 [0 0 0 0 0 0 3 0 0 2 3 0]
 [0 0 0 0 0 0 0 3 0 0 2 3]
 [0 0 0 0 0 0 0 0 3 0 3 2]]
'''
'''
min_nr=4, min_dis=0.01
bboxes: [[-1.000e-02 -1.000e-02  1.000e-02  1.000e-02]
 [-1.000e-02  9.000e-02  1.000e-02  1.100e-01]
 [ 9.000e-02 -1.000e-02  1.100e-01  1.000e-02]
 [ 9.000e-02  9.000e-02  1.100e-01  1.100e-01]
 [ 9.990e+00  9.990e+00  1.001e+01  1.001e+01]
 [ 9.990e+00  1.009e+01  1.001e+01  1.011e+01]
 [ 1.009e+01  9.990e+00  1.011e+01  1.001e+01]
 [ 1.009e+01  1.009e+01  1.011e+01  1.011e+01]]
labels [1 1 1 1 1 1 1 1]
Total edge number: 33

matrix: [[2 3 3 3 0 0 0 0]
 [3 2 3 3 0 0 0 0]
 [3 3 2 3 0 0 0 0]
 [3 3 3 2 0 0 0 0]
 [0 0 0 3 2 3 3 3]
 [0 0 0 0 3 2 3 3]
 [0 0 0 0 3 3 2 3]
 [0 0 0 0 3 3 3 2]]
'''