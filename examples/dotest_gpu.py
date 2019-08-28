#coding=utf-8
import tensorflow as tf
from wtfop.wtfop_ops import boxes_soft_nms,crop_boxes,boxes_encode,decode_boxes1,boxes_encode1,int_hash
import wtfop.wtfop_ops as wop
import object_detection.npod_toolkit as npod
import numpy as np
import wml_utils as wmlu
import random
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

class WTFOPTest(tf.test.TestCase):

    def testEncodeBoxes(self):
        with self.test_session() as sess:
            #人工核算
            np_gboxes = np.array([[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.5, 0.6], [0.1, 0.1, 0.4, 0.4], [0.7, 0.7, 0.9, 0.8]]);
            np_labels = np.array([1,2,3,4])
            np_boxes = np.array([[0.0, 0.0, 0.2, 0.1], [0.0, 0.0, 0.2, 0.2], [0.101, 0.1, 0.44, 0.4], [0.73, 0.71, 0.91, 0.81],
             [0.7, 0.1, 0.9, 0.5], [0.3, 0.481, 0.5, 0.7]]);
            np_lens = np.array([np_labels.shape[0]])
            gboxes = tf.constant(np_gboxes,dtype=tf.float32)
            glabels = tf.constant(np_labels);
            boxes = tf.constant(np_boxes,dtype=tf.float32)
            lens = tf.constant(np_lens,dtype=tf.int32)
            out_boxes, out_labels, out_scores, out_remove_indices,indices = boxes_encode(tf.expand_dims(boxes,0),
                                                                                 tf.expand_dims(gboxes,0),
                                                                                 tf.expand_dims(glabels,0),
                                                                                 length=lens,
                                                                                 pos_threshold=0.7,
                                                                                 neg_threshold=0.3,
                                                                                 prio_scaling=[0.1, 0.1, 0.2, 0.2])
            out_boxes, out_labels, out_scores, out_remove_indices,out_indices = sess.run([out_boxes, out_labels, out_scores, out_remove_indices,indices])
            target_out_boxes = np.array([[[0.,0.,0.,0.],
                  [0.,0.,0.,0.],
                  [-0.6047199,0.,-0.61108774,0.],
                  [-1.1111111,-1.,0.5268025782891318,0.],
                  [0.,0.,0.,0.],
                  [0.,-6.4155245,0.,1.573553724198501]]])
            target_out_labels = np.array([[0,1,3,4,0,2]])
            target_out_indices = np.array([[-1,0,2,3,-1,1]])
            target_out_scores = np.array([[0.,1.,0.87941164,0.67400867,0.,0.29750007]])
            target_out_remove_indices = np.array([[True,False,False,False,False,False]])
            self.assertAllEqual(a=target_out_remove_indices,b=out_remove_indices)
            self.assertAllEqual(a=target_out_indices,b=out_indices)
            self.assertAllClose(a=target_out_boxes,b=out_boxes,atol=1e-4,rtol=0.)
            self.assertAllEqual(a=target_out_labels,b=out_labels)
            self.assertAllClose(a=target_out_scores,b=out_scores,atol=1e-5,rtol=0.)

    '''def testEncodeBoxes1(self):
        with self.test_session() as sess:
            #人工核算
            np_gboxes = np.array([[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.5, 0.6], [0.1, 0.1, 0.4, 0.4], [0.7, 0.7, 0.9, 0.8]]);
            np_labels = np.array([1,2,3,4])
            np_boxes = np.array([[0.0, 0.0, 0.2, 0.1], [0.0, 0.0, 0.2, 0.2], [0.101, 0.1, 0.44, 0.4], [0.73, 0.71, 0.91, 0.81],
                                 [0.7, 0.1, 0.9, 0.5], [0.3, 0.481, 0.5, 0.7]]);
            #与上一个测试相比，长度不一样
            np_lens = np.array([np_labels.shape[0]-1])
            gboxes = tf.constant(np_gboxes,dtype=tf.float32)
            glabels = tf.constant(np_labels);
            boxes = tf.constant(np_boxes,dtype=tf.float32)
            lens = tf.constant(np_lens,dtype=tf.int32)
            out_boxes, out_labels, out_scores, out_remove_indices,indices = boxes_encode(tf.expand_dims(boxes,0),
                                                                                 tf.expand_dims(gboxes,0),
                                                                                 tf.expand_dims(glabels,0),
                                                                                 length=lens,
                                                                                 pos_threshold=0.7,
                                                                                 neg_threshold=0.3,
                                                                                 prio_scaling=[0.1, 0.1, 0.2, 0.2])
            out_boxes, out_labels, out_scores, out_remove_indices,out_indices = sess.run([out_boxes, out_labels, out_scores, out_remove_indices,indices])
            target_out_boxes = np.array([[[0.,0.,0.,0.],
                                          [0.,0.,0.,0.],
                                          [-0.6047199,0.,-0.61108774,0.],
                                          [0.,0.,0.,0.],
                                          [0.,0.,0.,0.],
                                          [0.,-6.4155245,0.,1.573553724198501]]])
            target_out_labels = np.array([[0,1,3,0,0,2]])
            target_out_indices = np.array([[-1,0,2,-1,-1,1]])
            target_out_scores = np.array([[0.,1.,0.87941164,0,0.,0.29750007]])
            target_out_remove_indices = np.array([[True,False,False,False,False,False]])
            self.assertAllEqual(a=target_out_indices,b=out_indices)
            self.assertAllEqual(a=target_out_remove_indices,b=out_remove_indices)
            self.assertAllClose(a=target_out_boxes,b=out_boxes,atol=1e-4,rtol=0.)
            self.assertAllEqual(a=target_out_labels,b=out_labels)
            self.assertAllClose(a=target_out_scores,b=out_scores,atol=1e-5,rtol=0.)'''

if __name__ == "__main__":
    random.seed(int(time.time()))
    tf.test.main()
