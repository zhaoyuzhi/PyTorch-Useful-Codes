# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 09:26:12 2018
@author: zhaoyuzhi
"""

import tensorflow as tf
import numpy
import sys
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)   #get image dataset

## define function for CNN
def weight_variable(shape):
    # Truncated normal distribution function
    # shape is kernel size, insize and outsize
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride = [1,x_movement,y_movement,1]
    # must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')

def max_pool_22(x):
    # stride = [1,x_movement,y_movement,1]
    # must have strides[0] = strides[3] = 1
    # ksize(kernel size) = [1,length,height,1]
    return tf.nn.max_pool(x, ksize = [1,2,2,1],
                          strides = [1,2,2,1], padding='SAME')

## define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None,784])      #28*28
ys = tf.placeholder(tf.float32, [None,10])       #the true value of y
keep_prob = tf.placeholder(tf.float32)
# -1 as default number of images, 1 as gray image
x_image = tf.reshape(xs, [-1,28,28,1])
# print(x_image.shape) [n_sanmples,28,28,1]

## convolutional layer 1, kernel 5*5, insize 1, outsize 32
# the first/second parameter are the sizes of kernel, the third parameter
# is the channel of images, the fourth parameter is feature map
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = conv2d(x_image, W_conv1) + b_conv1     #outsize = batch*28*28*32
a_conv1 = tf.nn.relu(h_conv1)                    #outsize = batch*28*28*32
# max pooling layer 1
h_pool1 = max_pool_22(a_conv1)                   #outsize = batch*14*14*32
a_pool1 = tf.nn.relu(h_pool1)                    #outsize = batch*14*14*32
# convolutional layer 2
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = conv2d(a_pool1, W_conv2) + b_conv2     #outsize = batch*14*14*64
a_conv2 = tf.nn.relu(h_conv2)                    #outsize = batch*14*14*64
# max pooling layer 2
h_pool2 = max_pool_22(a_conv2)                   #outsize = batch*7*7*64
a_pool2 = tf.nn.relu(h_pool2)                    #outsize = batch*7*7*64
# flat
a_pool2_flat = tf.reshape(a_pool2, [-1,7*7*64])  #[batch,7,7,64] = [batch,7*7*64]
# fully connected layer 1
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.matmul(a_pool2_flat, W_fc1) + b_fc1
a_fc1 = tf.nn.relu(h_fc1)
# dropout for fc layer1
a_fc1_drop = tf.nn.dropout(a_fc1, keep_prob)
# fully connected layer 2
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
h_fc2 = tf.matmul(a_fc1_drop, W_fc2) + b_fc2
prediction = tf.nn.softmax(h_fc2)                #the prediction value of y

## define loss and accuracy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                     reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
testaccuracy = list(range(10))   #should be equal to test times

## start training
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # train 1000 times
    for i in range(1000):
        # get 100 batch images every time
        batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs:batch[0], ys:batch[1], keep_prob:0.5})
    # test trained model
    for j in range(10):
        # get 100 batch images every time
        batch2 = mnist.test.next_batch(100)
        testaccuracy[j] = accuracy.eval(feed_dict={xs:batch2[0], ys:batch2[1], keep_prob:1.0})
        testaccuracy[j] = testaccuracy[j] * 100
    print("Test accuracy is:", end='')   #average accuracy
    sys.stdout.softspace=0   #delete 'space'
    print(numpy.mean(testaccuracy), end='')
    sys.stdout.softspace=0   #delete 'space'
    print("%")
    