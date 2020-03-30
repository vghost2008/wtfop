#coding=utf-8
import tensorflow as tf
import img_utils as wmli
import numpy as np
import wtfop.wtfop_ops as wop
import image_visualization as imv
import cv2
import wml_utils as wmlu


mask = np.array([[[1,1],[1,1]],[[1,0],[1,0]],[[1,0],[1,0]]])
boxes = np.array([[0,0,1,0.5],[0,0,0.5,0],[0,0,0.5,1]])
mask = tf.constant(mask,dtype=tf.float32)
boxes = tf.constant(boxes,dtype=tf.float32)
mask = wop.full_size_mask(mask=mask,bboxes=boxes,size=[4,4])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
res = sess.run(mask)
print(res)
wmlu.show_nparray(res)

'''
for i in range(6):
    image = img[i]
    image = imv.draw_polygon(image,res[i].astype(np.int32))
    cv2.imshow(f"img{i}",image)
cv2.waitKey(0)'''

