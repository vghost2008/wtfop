#coding=utf-8
import tensorflow as tf
import wtfop.wtfop_ops as wop
import numpy as np
import random
import time
import os
import wml_utils as wmlu
import object_detection2.bboxes as odb
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
slim = tf.contrib.slim
dev = ":/gpu:0"

class WTFOPTest(tf.test.TestCase):

    def test_left_pool(self):
        with self.test_session() as sess:
            data = np.array([[1,2,1],[5,3,4]],dtype=np.float32)
            data = tf.constant(np.reshape(data,[1,2,3,1]))
            pooled = wop.left_pool(data)
            pooled = tf.squeeze(pooled,axis=3)
            pooled = tf.squeeze(pooled,axis=0)
            y = tf.reduce_sum(pooled)
            grad = tf.gradients(y,data)
            grad = tf.squeeze(grad,axis=0)
            grad = tf.squeeze(grad,axis=3)
            grad = tf.squeeze(grad,axis=0)
            target_pooled = np.array([[2.0,2.0,1.0],
                                        [5.0,4.0,4.0]])
            target_grad = np.array([[0.0,2.0,1.0],
                                    [1.0,0.0,2.0]])
            pooled = sess.run(pooled)
            grad = sess.run(grad)
            self.assertAllClose(pooled,target_pooled,atol=1e-4)
            self.assertAllClose(grad,target_grad,atol=1e-4)

    def test_right_pool(self):
        with self.test_session() as sess:
            data = np.array([[1,2,1],[4,3,5]],dtype=np.float32)
            data = tf.constant(np.reshape(data,[1,2,3,1]))
            pooled = wop.right_pool(data)
            pooled = tf.squeeze(pooled,axis=3)
            pooled = tf.squeeze(pooled,axis=0)
            y = tf.reduce_sum(pooled)
            grad = tf.gradients(y,data)
            grad = tf.squeeze(grad,axis=0)
            grad = tf.squeeze(grad,axis=3)
            grad = tf.squeeze(grad,axis=0)
            target_pooled = np.array([[1.0,2.0,2.0],
                                        [4.0,4.0,5]])
            target_grad = np.array([[1.0,2.0,0.0],
                                    [2.0,0.0,1.0]])
            pooled = sess.run(pooled)
            grad = sess.run(grad)
            self.assertAllClose(pooled,target_pooled,atol=1e-4)
            self.assertAllClose(grad,target_grad,atol=1e-4)

    def test_bottom_pool(self):
        with self.test_session() as sess:
            data = np.array([[1,2,1],[4,3,5]],dtype=np.float32)
            data = np.transpose(data,axes=[1,0])
            data = tf.constant(np.reshape(data,[1,3,2,1]))
            pooled = wop.bottom_pool(data)
            pooled = tf.squeeze(pooled,axis=3)
            pooled = tf.squeeze(pooled,axis=0)
            y = tf.reduce_sum(pooled)
            grad = tf.gradients(y,data)
            grad = tf.squeeze(grad,axis=0)
            grad = tf.squeeze(grad,axis=3)
            grad = tf.squeeze(grad,axis=0)
            target_pooled = np.array([[1.0,2.0,2.0],
                                        [4.0,4.0,5]])
            target_pooled = np.transpose(target_pooled,axes=[1,0])
            target_grad = np.array([[1.0,2.0,0.0],
                                    [2.0,0.0,1.0]])
            target_grad = np.transpose(target_grad,axes=[1,0])
            pooled = sess.run(pooled)
            grad = sess.run(grad)
            self.assertAllClose(pooled,target_pooled,atol=1e-4)
            self.assertAllClose(grad,target_grad,atol=1e-4)

    def test_top_pool(self):
        with self.test_session() as sess:
            data = np.array([[1,2,1],[5,3,4]],dtype=np.float32)
            data = np.transpose(data,axes=[1,0])
            data = tf.constant(np.reshape(data,[1,3,2,1]))
            pooled = wop.top_pool(data)
            pooled = tf.squeeze(pooled,axis=3)
            pooled = tf.squeeze(pooled,axis=0)
            y = tf.reduce_sum(pooled)
            grad = tf.gradients(y,data)
            grad = tf.squeeze(grad,axis=0)
            grad = tf.squeeze(grad,axis=3)
            grad = tf.squeeze(grad,axis=0)
            target_pooled = np.array([[2.0,2.0,1.0],
                                      [5.0,4.0,4.0]])
            target_pooled = np.transpose(target_pooled,axes=[1,0])
            target_grad = np.array([[0.0,2.0,1.0],
                                    [1.0,0.0,2.0]])
            target_grad = np.transpose(target_grad,axes=[1,0])
            pooled = sess.run(pooled)
            grad = sess.run(grad)
            self.assertAllClose(pooled,target_pooled,atol=1e-4)
            self.assertAllClose(grad,target_grad,atol=1e-4)


if __name__ == "__main__":
    random.seed(int(time.time()))
    tf.test.main()
