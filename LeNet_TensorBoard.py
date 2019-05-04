# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:00:54 2018
@author: zhaoyuzhi
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)   #get image dataset

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

myGraph = tf.Graph()
with myGraph.as_default():
    with tf.name_scope('inputsAndLabels'):
        xs = tf.placeholder(tf.float32, shape=[None, 784])
        ys = tf.placeholder(tf.float32, shape=[None, 10])

    with tf.name_scope('conv1/pool1'):
        x_image = tf.reshape(xs, shape=[-1,28,28,1])
        # convolutional layer 1
        W_conv1 = weight_variable([5,5,1,32])
        b_conv1 = bias_variable([32])
        h_conv1 = conv2d(x_image, W_conv1) + b_conv1     #outsize = batch*28*28*32
        a_conv1 = tf.nn.relu(h_conv1)                    #outsize = batch*28*28*32
        # max pooling layer 1
        h_pool1 = max_pool_22(a_conv1)                   #outsize = batch*14*14*32
        a_pool1 = tf.nn.relu(h_pool1)                    #outsize = batch*14*14*32

        tf.summary.image('x_input', xs, max_outputs=10)
        tf.summary.histogram('W_con1', W_conv1)
        tf.summary.histogram('b_con1', b_conv1)

    with tf.name_scope('conv2/pool2'):
        # convolutional layer 2
        W_conv2 = weight_variable([5,5,32,64])
        b_conv2 = bias_variable([64])
        h_conv2 = conv2d(a_pool1, W_conv2) + b_conv2     #outsize = batch*14*14*64
        a_conv2 = tf.nn.relu(h_conv2)                    #outsize = batch*14*14*64
        # max pooling layer 2
        h_pool2 = max_pool_22(a_conv2)                   #outsize = batch*7*7*64
        a_pool2 = tf.nn.relu(h_pool2)                    #outsize = batch*7*7*64

        tf.summary.histogram('W_con2', W_conv2)
        tf.summary.histogram('b_con2', b_conv2)

    with tf.name_scope('fc1'):
        # flat
        a_pool2_flat = tf.reshape(a_pool2, [-1,7*7*64])  #[batch,7,7,64] = [batch,7*7*64]
        # fully connected layer 1
        W_fc1 = weight_variable([7*7*64,1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.matmul(a_pool2_flat, W_fc1) + b_fc1
        a_fc1 = tf.nn.relu(h_fc1)
        keep_prob = tf.placeholder(tf.float32)
        a_fc1_drop = tf.nn.dropout(a_fc1, keep_prob)

        tf.summary.histogram('W_fc1', W_fc1)
        tf.summary.histogram('b_fc1', b_fc1)

    with tf.name_scope('fc2'):
        # fully connected layer 2
        W_fc2 = weight_variable([1024,10])
        b_fc2 = bias_variable([10])
        h_fc2 = tf.matmul(a_fc1_drop, W_fc2) + b_fc2
        prediction = tf.nn.softmax(h_fc2)                #the prediction value of y

        tf.summary.histogram('W_fc2', W_fc2)
        tf.summary.histogram('b_fc2', b_fc2)

    with tf.name_scope('train'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('loss', cross_entropy)
        tf.summary.scalar('accuracy', accuracy)


with tf.Session(graph=myGraph) as sess:
    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./mnistEven/', graph=sess.graph)

    for i in range(10001):
        batch = mnist.train.next_batch(50)
        sess.run(train_step,feed_dict={xs:batch[0], ys:batch[1], keep_prob:0.5})
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={xs:batch[0], ys:batch[1], keep_prob:1.0})
            print('step %d training accuracy:%g'%(i, train_accuracy))

            summary = sess.run(merged,feed_dict={xs:batch[0], ys:batch[1], keep_prob:1.0})
            summary_writer.add_summary(summary,i)

    test_accuracy = accuracy.eval(feed_dict={xs:mnist.test.images, ys:mnist.test.labels, keep_prob:1.0})
    print('test accuracy:%g' %test_accuracy)
    