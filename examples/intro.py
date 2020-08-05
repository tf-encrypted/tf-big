import tensorflow as tf
import tf_big

# load large values as strings
x = tf_big.constant([["100000000000000000000", "200000000000000000000"]])

# load ordinary TensorFlow tensors
y = tf_big.import_tensor(tf.constant([[3, 4]]))

# perform computation as usual
z = x * y

# export result back into a TensorFlow tensor
tf_res = tf_big.export_tensor(z)
print(tf_res)
