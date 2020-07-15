#coding=utf-8
import tensorflow as tf
from wtfop.wtfop_ops import fcos_boxes_encode
import object_detection2.bboxes as odbox
import wml_utils as wmlu
import wtfop.wtfop_ops as wop
import wml_tfutils as wmlt
import numpy as np
import random
import time
import os
#import matplotlib.pyplot as plt
import object_detection2.bboxes as odb
os.environ['CUDA_VISIBLE_DEVICES'] = ''
slim = tf.contrib.slim
dev = ":/gpu:0"

class WTFOPTest(tf.test.TestCase):

    def apply_deltas(self, regression, img_size=None, fm_size=None):
        if len(regression.get_shape()) == 2:
            B = 1
            H = fm_size[0]
            W = fm_size[1]

        elif len(regression.get_shape()) == 4:
            B, H, W, _ = wmlt.combined_static_and_dynamic_shape(regression)
        else:
            raise NotImplementedError("Error")
        x_i, y_i = tf.meshgrid(tf.range(W), tf.range(H))
        if isinstance(img_size, tf.Tensor) and img_size.dtype != tf.float32:
            img_size = tf.to_float(img_size)
        H = tf.to_float(H)
        W = tf.to_float(W)
        y_f = tf.to_float(y_i) + 0.5
        x_f = tf.to_float(x_i) + 0.5
        y_delta = img_size[0] / H
        x_delta = img_size[1] / W
        y_base_value = y_f * y_delta
        x_base_value = x_f * x_delta
        base_value = tf.stack([y_base_value, x_base_value, y_base_value, x_base_value], axis=-1)
        if len(regression.get_shape()) == 4:
            base_value = tf.expand_dims(base_value, axis=0)
            base_value = tf.stop_gradient(tf.tile(base_value, [B, 1, 1, 1]))
            multi = tf.convert_to_tensor([[[[-1, -1, 1, 1]]]], dtype=tf.float32)
        elif len(regression.get_shape()) == 2:
            base_value = tf.reshape(base_value, [-1, 4])
            multi = tf.convert_to_tensor([[-1, -1, 1, 1]], dtype=tf.float32)

        return base_value + regression * multi

    def test_fcos_boxes_encode(self):
        box = np.array([[[0.25,0.25,0.75,0.75],[0.1,0.1,0.3,0.3]]],dtype=np.float32)
        labels = np.array([[1,2]],dtype=np.int32)
        length = np.array([2],dtype=np.int32)
        fm_shape = [32,16]
        img_size = [224,112]
        with self.test_session() as sess:
            regression,center_ness,gt_boxes,classes = fcos_boxes_encode(min_size=0,
                                                                        max_size=128,
                                                                        fm_shape=fm_shape,
                                                                        img_size=img_size,
                                                                        gbboxes=box,
                                                                        glabels=labels,
                                                                        glength=length)
            regression = wmlt.PrintSummary(regression,"regression")
            boxes = self.apply_deltas(regression,img_size=img_size)
            mask = tf.greater(classes,0)
            boxes = tf.boolean_mask(boxes,mask)
            boxes = odbox.tfabsolutely_boxes_to_relative_boxes(boxes,width=tf.convert_to_tensor(img_size[1]),
                                                               height=tf.convert_to_tensor(img_size[0]))
            regression = tf.Print(regression,["R",regression],summarize=1000)
            regression = tf.Print(regression,["C",classes],summarize=1000)
            regression, center_ness, gt_boxes, classes,boxes = sess.run([regression,center_ness,gt_boxes,classes,boxes])
            #plt.figure(figsize=(10,10))
            #plt.imshow(regression[0,:,:,3])
            #plt.imshow(gt_boxes[0,:,:,1])
            #plt.imshow(classes[0,:,:])
            #plt.show()

            print(regression, center_ness, gt_boxes, classes)
            wmlu.show_nparray(boxes)


if __name__ == "__main__":
    random.seed(int(time.time()))
    tf.test.main()
