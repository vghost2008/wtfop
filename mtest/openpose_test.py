import wtfop.wtfop_ops as wop
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

l_delta = 8
gaussian_delta = 5
keypoints_pair = [0,2,2,1,1,3]
keypoints = [0.3,0.3,0.6,-1,0.6,0.3,0.3,0.6]
keypoints = np.array(keypoints,dtype=np.float32)
keypoints = np.reshape(keypoints,[1,1,4,2])
output_size = np.array([100,100],dtype=np.int32)
glength = np.array([1],dtype=np.int32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
output = wop.hr_net_encode(keypoints,output_size,glength,gaussian_delta)
conf_map,indexs = sess.run(output)
print(indexs)
for i in range(conf_map.shape[-1]):
	plt.figure(i,(10,10))
	img = conf_map[0,:,:,i]*255
	print(i,np.reshape(conf_map[0],[-1])[23723])
	plt.imshow(img)
plt.show()

