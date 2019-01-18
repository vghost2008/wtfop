#coding=utf-8
import tensorflow as tf
from wtfop.wtfop_ops import boxes_soft_nms

class SquareTest(tf.test.TestCase):
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


if __name__ == "__main__":
    tf.test.main()
