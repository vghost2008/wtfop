#coding=utf-8
import tensorflow as tf
from wtfop.wtfop_ops import boxes_soft_nms,crop_boxes,boxes_encode,decode_boxes1,boxes_encode1,label_type
import object_detection.npod_toolkit as npod
import numpy as np
import random
import time

class WTFOPTest(tf.test.TestCase):

    @staticmethod
    def get_random_box(w=1.0,h=1.0):
        min_y = random.random()
        min_x = random.random()
        max_y = min_y+random.random()*(1.0-min_y)
        max_x = min_x+random.random()*(1.0-min_x)
        return (min_y*h,min_x*w,max_y*h,max_x*w)

    def testSoftNMS(self):
        with self.test_session() as sess:
            boxes=tf.constant([[124,60,251,153],[161,85,293,193],[104,103,266,222],[277,371,414,484]],dtype=tf.float32)
            labels = tf.constant([1,1,1,1],dtype=tf.int32)
            probs = tf.constant([0.9,0.8,0.8,0.9],dtype=tf.float32)
            boxes, labels, indices = boxes_soft_nms(boxes, labels, confidence=probs,
                                                    threshold=0.7,
                                                    classes_wise=True)
            boxes,labels,indices = sess.run([boxes,labels,indices])
            print(boxes)
            print(labels)
            print(indices)
            self.assertAllEqual(a=indices,b=[0,2,3],msg="index equall")
    def testCropBoxes(self):
        with self.test_session() as sess:
            #np_gboxes = np.array([[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.5, 0.6], [0.1, 0.1, 0.4, 0.4], [0.7, 0.7, 0.9, 0.8]])
            #np_subbox = np.array([0.1, 0.1, 0.7, 0.7])
            np_gboxes_nr = 128
            np_gboxes = []
            for _ in range(np_gboxes_nr):
                np_gboxes.append(self.get_random_box())
            np_gboxes = np.array(np_gboxes)
            np_subbox = self.get_random_box()
            np_res_boxes,np_res_mask = npod.crop_box(bboxes=np_gboxes, sub_box=np_subbox, remove_threshold=0.8)
            gboxes = tf.constant(np_gboxes, dtype=tf.float32);
            ref_boxes = tf.constant(np_subbox, dtype=tf.float32);
            boxes, mask = crop_boxes(ref_box=ref_boxes, boxes=gboxes, threshold=0.8)
            res_boxes,res_mask = sess.run([boxes,mask])
            self.assertAllEqual(a=np_res_mask,b=res_mask)
            self.assertAllClose(a=np_res_boxes,b=res_boxes,atol=1e-6,rtol=0.)

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
            out_boxes, out_labels, out_scores, out_remove_indices = boxes_encode(tf.expand_dims(boxes,0),
                                                                                 tf.expand_dims(gboxes,0),
                                                                                 tf.expand_dims(glabels,0),
                                                                                 length=lens,
                                                                                 pos_threshold=0.7,
                                                                                 neg_threshold=0.3,
                                                                                 prio_scaling=[0.1, 0.1, 0.2, 0.2])
            out_boxes, out_labels, out_scores, out_remove_indices = sess.run([out_boxes, out_labels, out_scores, out_remove_indices])
            target_out_boxes = np.array([[[0.,0.,0.,0.],
                  [0.,0.,0.,0.],
                  [-0.6047199,0.,-0.61108774,0.],
                  [-1.1111111,-1.,0.5268025782891318,0.],
                  [0.,0.,0.,0.],
                  [0.,-6.4155245,0.,1.573553724198501]]])
            target_out_labels = np.array([[0,1,3,4,0,2]])
            target_out_scores = np.array([[0.,1.,0.87941164,0.67400867,0.,0.29750007]])
            target_out_remove_indices = np.array([[True,False,False,False,False,False]])
            self.assertAllEqual(a=target_out_remove_indices,b=out_remove_indices)
            self.assertAllClose(a=target_out_boxes,b=out_boxes,atol=1e-4,rtol=0.)
            self.assertAllEqual(a=target_out_labels,b=out_labels)
            self.assertAllClose(a=target_out_scores,b=out_scores,atol=1e-5,rtol=0.)

    def testEncodeBoxes1(self):
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
            out_boxes, out_labels, out_scores, out_remove_indices = boxes_encode(tf.expand_dims(boxes,0),
                                                                                 tf.expand_dims(gboxes,0),
                                                                                 tf.expand_dims(glabels,0),
                                                                                 length=lens,
                                                                                 pos_threshold=0.7,
                                                                                 neg_threshold=0.3,
                                                                                 prio_scaling=[0.1, 0.1, 0.2, 0.2])
            out_boxes, out_labels, out_scores, out_remove_indices = sess.run([out_boxes, out_labels, out_scores, out_remove_indices])
            target_out_boxes = np.array([[[0.,0.,0.,0.],
                                          [0.,0.,0.,0.],
                                          [-0.6047199,0.,-0.61108774,0.],
                                          [0.,0.,0.,0.],
                                          [0.,0.,0.,0.],
                                          [0.,-6.4155245,0.,1.573553724198501]]])
            target_out_labels = np.array([[0,1,3,0,0,2]])
            target_out_scores = np.array([[0.,1.,0.87941164,0,0.,0.29750007]])
            target_out_remove_indices = np.array([[True,False,False,False,False,False]])
            self.assertAllEqual(a=target_out_remove_indices,b=out_remove_indices)
            self.assertAllClose(a=target_out_boxes,b=out_boxes,atol=1e-4,rtol=0.)
            self.assertAllEqual(a=target_out_labels,b=out_labels)
            self.assertAllClose(a=target_out_scores,b=out_scores,atol=1e-5,rtol=0.)

    def testDecodeBoxes(self):
        with self.test_session() as sess:
            #人工核算
            np_gboxes = np.array([[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.5, 0.6], [0.1, 0.1, 0.4, 0.4], [0.7, 0.7, 0.9, 0.8]]);
            np_labels = np.array([1,2,3,4])
            np_boxes = np.array([[0.0, 0.0, 0.2, 0.1], [0.0, 0.0, 0.2, 0.2], [0.101, 0.1, 0.44, 0.4], [0.73, 0.71, 0.91, 0.81],
                                 [0.7, 0.1, 0.9, 0.5], [0.3, 0.481, 0.5, 0.7]])
            gboxes = tf.constant(np_gboxes,dtype=tf.float32)
            glabels = tf.constant(np_labels);
            boxes = tf.constant(np_boxes,dtype=tf.float32)
            out_boxes, out_labels, out_scores, out_remove_indices = boxes_encode1(boxes,
                                                                                 gboxes,
                                                                                 glabels,
                                                                                 pos_threshold=0.7,
                                                                                 neg_threshold=0.3,
                                                                                 prio_scaling=[0.1, 0.1, 0.2, 0.2])
            out_boxes, out_labels, out_scores, out_remove_indices = sess.run([out_boxes, out_labels, out_scores, out_remove_indices])
            target_out_boxes = np.array([[0.,0.,0.,0.],
                                          [0.,0.,0.,0.],
                                          [-0.6047199,0.,-0.61108774,0.],
                                          [-1.1111111,-1.,0.5268025782891318,0.],
                                          [0.,0.,0.,0.],
                                          [0.,-6.4155245,0.,1.573553724198501]])
            target_out_labels = np.array([0,1,3,4,0,2])
            target_out_scores = np.array([0.,1.,0.87941164,0.67400867,0.,0.29750007])
            target_out_remove_indices = np.array([True,False,False,False,False,False])
            self.assertAllEqual(a=target_out_remove_indices,b=out_remove_indices)
            self.assertAllClose(a=target_out_boxes,b=out_boxes,atol=1e-4,rtol=0.)
            self.assertAllEqual(a=target_out_labels,b=out_labels)
            self.assertAllClose(a=target_out_scores,b=out_scores,atol=1e-5,rtol=0.)
            keep_indices = tf.logical_and(tf.logical_not(out_remove_indices),tf.greater(out_scores,0.1))
            boxes = tf.boolean_mask(boxes,keep_indices)
            out_boxes = tf.boolean_mask(out_boxes,keep_indices)

            new_boxes = decode_boxes1(boxes,out_boxes)
            out_new_boxes= new_boxes.eval()
            target_new_boxes = np.array([[0.,0.,0.2,0.2],
                                         [0.09999999,0.09999999,0.4,0.4],
                                         [0.6999999,0.7,0.9,0.8],
                                         [0.3,0.3,0.5,0.6]])
            self.assertAllClose(a=out_new_boxes,b=target_new_boxes,atol=1e-5,rtol=0.)

    def testLabelType(self):
        text = []
        for i in range(ord('a'), ord('z') + 1):
            text.append(chr(i))
        for i in range(ord('A'), ord('Z') + 1):
            text.append(chr(i))
        for i in range(ord('0'), ord('9') + 1):
            text.append(chr(i))
        text.append('/')
        text.append('\\')
        text.append('-')
        text.append('+')
        text.append(":")
        text.append("WORD")
        text_to_id = {}
        for i, t in enumerate(text):
            text_to_id[t] = i + 1
        def string_to_ids(v):
            res = []
            for c in v:
                res.append(text_to_id[c])
            return res
        def make_bboxes(ids):
            w = 1.
            h  = 2.;
            res = []
            for i in range(len(ids)):
                res.append([0.,w*i,h,w*(i+1)])
            res.append([0.,0.,h,w*(len(ids))])
            return np.array(res)
        test_data=[
           "Ki-67","kI-67","ER","er","Her-2","HER-2","HP","hp",
            "k-67","eir","hr-","hhpp",
        ]
        expected_data=[0,0,1,1,2,2,3,3,0,1,2,3]
        t_bboxes = tf.placeholder(dtype=tf.float32,shape=[None,4])
        t_labels = tf.placeholder(dtype=tf.int32,shape=[None])
        t_type = label_type(bboxes=t_bboxes,labels=t_labels)
        with self.test_session() as sess:
            for i,data in enumerate(test_data):
                print(i)
                ids = string_to_ids(data)
                bboxes = make_bboxes(ids)
                ids.append(68)
                type = sess.run(t_type,feed_dict={t_bboxes:bboxes,t_labels:ids})
                print(test_data[i],type)
                self.assertAllEqual(type,np.array([expected_data[i]]))


if __name__ == "__main__":
    random.seed(int(time.time()))
    tf.test.main()