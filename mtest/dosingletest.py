#coding=utf-8
import tensorflow as tf
import img_utils as wmli
import numpy as np
import wtfop.wtfop_ops as wop
import image_visualization as imv
import cv2
import wml_utils as wmlu

img_files = []
for i in range(6):
    img_files.append(f"./imgs/img{i}.jpg")
img = []
for f in img_files:
    img.append(wmli.nprgb_to_gray(wmli.imread(f)).astype(np.uint8))
imgs = np.stack(img,axis=0)

res = wop.min_area_rect(imgs,res_points=False)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
res = sess.run(res)
print(res)
wmlu.show_nparray(res)

'''
for i in range(6):
    image = img[i]
    image = imv.draw_polygon(image,res[i].astype(np.int32))
    cv2.imshow(f"img{i}",image)
cv2.waitKey(0)'''

