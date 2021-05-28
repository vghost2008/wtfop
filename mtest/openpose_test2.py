import wtfop.wtfop_ops as wop
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

l_delta = 2
gaussian_delta = 8
keypoints_pair = [0,1,1,2,2,0,1,3,2,4,0,5,0,6,5,6,5,11,6,12,11,12,5,7,7,9,6,8,8,10,11,13,13,15,12,14,14,16]
x = [0.6776212832550861,0.6791862284820032,-1.0,0.6932707355242567,-1.0,0.6572769953051644,0.7417840375586855,0.6197183098591549,0.7652582159624414,-1.0,-1.0,0.6557120500782473,0.7167449139280125,0.6431924882629108,0.7167449139280125,0.6291079812206573,0.7276995305164319,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]
y = [0.22065727699530516,0.2112676056338028,-1.0,0.2300469483568075,-1.0,0.3004694835680751,0.31220657276995306,0.38028169014084506,0.4061032863849765,-1.0,-1.0,0.5023474178403756,0.5046948356807511,0.6431924882629108,0.6408450704225352,0.7816901408450704,0.784037558685446,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]
x = np.array(x)
y = np.array(y)
x = np.reshape(x,[-1,17,1])
y = np.reshape(y,[-1,17,1])
keypoints = np.stack([x,y],axis=-1)
keypoints = np.array(keypoints,dtype=np.float32)
keypoints = np.reshape(keypoints,[1,-1,17,2])
output_size = np.array([64,64],dtype=np.int32)
glength = np.array([keypoints.shape[1]],dtype=np.int32)

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
