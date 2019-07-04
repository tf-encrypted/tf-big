import tensorflow as tf
import tf_big

x = tf_big.Tensor([[1,2,3,4]])
y = tf_big.Tensor([[1,2,3,4]])
z = x + y

with tf.Session() as sess:
  res = z.eval(session=sess)
  print(res)