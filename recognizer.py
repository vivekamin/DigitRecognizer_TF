import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32 , [None, 784])  #input: image vectors
W = tf.Variable(tf.Zeros([784,10]))
b = tf.Variable(tf.Zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

