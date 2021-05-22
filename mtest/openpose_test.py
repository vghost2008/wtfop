import wtfop.wtfop_ops as wop
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

l_delta = 1
gaussian_delta = 5
keypoints_pair = [0,2,2,1,1,3]
keypoints = [0.3,0.3,0.6,0.6,0.6,0.3,0.3,0.6]
keypoints = np.array(keypoints,dtype=np.float32)
keypoints = np.reshape(keypoints,[1,1,4,2])
output_size = np.array([100,100],dtype=np.int32)
glength = np.array([1],dtype=np.int32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
output = wop.open_pose_encode(keypoints,output_size,glength,keypoints_pair,l_delta,gaussian_delta)
conf_map,paf_map= sess.run(output)
for i in range(conf_map.shape[-1]):
	plt.figure(i,(10,10))
	img = conf_map[0,:,:,i]*255
	plt.imshow(img)
plt.show()

for i in range(paf_map.shape[-1]):
	plt.figure(i,(10,10))
	data = paf_map[0,:,:,i]
	print(i,np.min(data),np.mean(data),np.max(data))
	img = paf_map[0,:,:,i]*255
	plt.imshow(img,cmap="gray")
plt.show()
#def open_pose_decode(conf_maps,paf_maps,keypoints_pair,keypoints_th=0.1,interp_samples=10,paf_score_th=0.1,conf_th=0.7,max_detection=100):
output = wop.open_pose_decode(conf_map,paf_map,keypoints_pair,max_detection=2)
keypoints = sess.run(output)
print(keypoints)
