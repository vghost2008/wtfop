from __future__ import print_function

import tensorflow as tf
import numpy as np

from wtfop.wtfop_ops import boxes_encode1,decode_boxes1
sess = tf.Session()
gboxes = tf.Variable([[0.0,0.0,0.2,0.2],[0.3,0.3,0.5,0.6],[0.1,0.1,0.4,0.4],[0.7,0.7,0.9,0.8]]);
glabels = tf.Variable([1,2,3,4]);
boxes = tf.Variable([[0.0,0.0,0.2,0.1],[0.0,0.0,0.2,0.2],[0.101,0.1,0.44,0.4],[0.73,0.71,0.91,0.81],
[0.7,0.1,0.9,0.5],[0.3,0.481,0.5,0.7]]);
out_boxes,out_labels,out_scores,out_remove_indices = boxes_encode1(boxes,gboxes,glabels,0.7,0.3)
decoded_boxes = decode_boxes1(boxes,out_boxes)
sess.run(tf.global_variables_initializer())
print("gboxes:",sess.run(gboxes))
print("boxes",sess.run(boxes))
print("out_boxes:",sess.run(out_boxes))
print("out_labels",sess.run(out_labels))
print("out_scoures",sess.run(out_scores))
print("out_remove_indices",sess.run(out_remove_indices))
print("out_decoded_boxes",sess.run(decoded_boxes))
'''
gboxes: [[0.  0.  0.2 0.2]
 [0.3 0.3 0.5 0.6]
 [0.1 0.1 0.4 0.4]
 [0.7 0.7 0.9 0.8]]
boxes [[0.    0.    0.2   0.1  ]
 [0.    0.    0.2   0.2  ]
 [0.101 0.1   0.44  0.4  ]
 [0.73  0.71  0.91  0.81 ]
 [0.7   0.1   0.9   0.5  ]
 [0.3   0.481 0.5   0.7  ]]
out_boxes: [[ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [-0.6047199   0.         -0.61108774  0.        ]
 [-1.1111166  -0.9999988   0.52680224  0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.         -6.4155245   0.          1.5735543 ]]
out_labels [0 1 3 4 0 2]
out_scoures [0.         1.         0.87941164 0.67400867 0.         0.29750007]
out_remove_indices [ True False False False False False]
out_decoded_boxes [[0.         0.         0.2        0.1       ]
 [0.         0.         0.2        0.2       ]
 [0.09999999 0.09999999 0.4        0.4       ]
 [0.6999999  0.7        0.9        0.8       ]
 [0.6999999  0.10000001 0.9        0.5       ]
 [0.3        0.3        0.5        0.6       ]]
'''
