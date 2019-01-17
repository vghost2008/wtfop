#coding=utf-8
import tensorflow as tf
from wtfop.wtfop_ops import boxes_soft_nms
import PyQt4Enhanced

class SquareTest(tf.test.TestCase):
    def testSparseSoftmaxCrossEntropyWithLogitsFL(self):
        with self.test_session() as sess:
            shape = [2,3,4,5]
            tf.set_random_seed(int(time.time()))
            logits = tf.random_uniform(shape=shape,minval=-9.,maxval=9.,dtype=tf.float32)
            labels = tf.random_uniform(shape=shape[:-1],minval=0,maxval=shape[-1],dtype=tf.int32)
            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
            loss2= wnn.sparse_softmax_cross_entropy_with_logits_FL(labels=labels,logits=logits,gamma=0.)
            t_loss1,t_loss2,t_labels = sess.run([loss1,loss2,labels])
            print(t_labels)


            self.assertAllClose(a=t_loss1,b=t_loss2,atol=0.01,rtol=0)




if __name__ == "__main__":
    tf.test.main()
