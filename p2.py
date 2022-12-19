#find eigen value vector

import tensorflow as tf

a = tf.constant([1,2,3,4,5,6], shape = [2,3])
b = tf.constant([7,8,9,10,11,12], shape = [3,2])
x = tf.matmul(a,b)
print(x)

ematrix = tf.random.uniform([2,2], minval=3, maxval=10, dtype=tf.float32, name="Matrix A")
print("Matrix A {}".format(ematrix))
evalue , evector = tf.linalg.eigh(ematrix)
print("Eigen Value {} \n Eigen Vector{}".format(evalue, evector))