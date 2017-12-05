# -*- coding: utf-8 -*-
import os
import glob
from sys import argv
from PIL import Image,ImageFilter
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#read MNIST data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()

#MNIST data input, img size: 28*28 = 784
#number of classes = 10
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

def weight(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                        strides = [1, 2, 2, 1], padding='SAME')

#first convolution layer
#5*5 is the size of receptive field
w_conv1 = weight([5, 5, 1, 32])
b_conv1 = bias([32])
x_image = tf.reshape(x, [-1,28,28,1])

conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
#max pooling
conv1 = max_pool_2x2(conv1)

#second convolution layer
w_conv2 = weight([5, 5, 32, 64])
b_conv2 = bias([64])

conv2 = tf.nn.relu(conv2d(conv1, w_conv2) + b_conv2)
#max pooling
conv2 = max_pool_2x2(conv2)
#output_size:7*7*64

#fully connected layer 1
w_fc1 = weight([7 * 7 * 64, 1024])
b_fc1 = bias([1024])

fc1 = tf.reshape(conv2, [-1, 7*7*64])
fc1 = tf.nn.relu(tf.matmul(fc1, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
fc1 = tf.nn.dropout(fc1, keep_prob)

#fully connected layer 2
w_fc2 = weight([1024, 10])
b_fc2 = bias([10])

y_conv = tf.nn.softmax(tf.matmul(fc1, w_fc2) + b_fc2)

def train():

  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess.run(tf.initialize_all_variables())
  saver = tf.train.Saver()

  for i in range(30000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x:batch[0], y_: batch[1], keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  print("test accuracy %g"%accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

  saver.save(sess, os.path.join(os.getcwd(), 'trained_model.ckpt'))
  
def getPicArray(filename):

  im = Image.open(filename)
  out = im.resize((28, 28), Image.ANTIALIAS) 
  
  im_arr = np.array(out.convert('L'))
  
  num0 = 0
  num255 = 0
  threshold = 100

  for x in range(28):
      for y in range(28):
          if im_arr[x][y] > threshold : num255 = num255 + 1
          else : num0 = num0 + 1

  if(num255 > num0) :
          for x in range(28):
              for y in range(28):
                  im_arr[x][y] = 255 - im_arr[x][y]
                  if(im_arr[x][y] < threshold) :  im_arr[x][y] = 0

  nm = im_arr.reshape((1, 784))
  
  nm = nm.astype(np.float32)
  nm = np.multiply(nm, 1.0 / 255.0)
  
  return nm
      
def testImage(paths):

  saver = tf.train.Saver()
  saver.restore(sess, os.path.join(os.getcwd(), 'trained_model.ckpt'))
  
  ann = getStandard()
  accuracy = 0.0

  f = open('predictions.txt', 'w')
  
  for path in paths:
    oneTestx = getPicArray(path)
    predict = sess.run(tf.argmax(y_conv, 1), feed_dict = {x:oneTestx, keep_prob:1})

    file = path.split("\\")[1] 
    f.write(file + "	" + str(predict[0]) + '\n')
    if str(predict[0]) == ann[file]:
      accuracy += 1

  f.close()

  print("Accuracy: %g"%(accuracy/195.0))

def getStandard():

  ann = {}
  with open("annotation.txt") as f:
    content = f.readlines()

  for line in content:
    line = line.strip().split("	")
    ann[line[0]] = line[1]

  return ann

if __name__ == '__main__':
  path = argv[1]
  path_list = glob.glob(path+'*png')
  #train()
  testImage(path_list)
