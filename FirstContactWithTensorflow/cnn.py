import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/", one_hot=True)

num_input = 784
num_class = 10

X = tf.placeholder(tf.float32, [None, num_input]) # 1개의 이미지는 28 X 28 = 784 픽셀
t = tf.placeholder(tf.float32, [None, num_class]) # 1개의 이미지에 대한 label은 10가지 클래스 중 하나 : One-hot encoding

x_images = tf.reshape(X, [-1, 28, 28, 1]) # CNN은 한개의 이미지를 3차원(폭, 높이, 채널)으로 다루기 때문에


# 초기화된 가중치 생성
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 0.1로 초기화된 편향 생성
def bias_variable(shape):
    initial = tf.constant(0.1, shape)
    return tf.Variable(initial)


# Conv 계층
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# Max Pool 계층: 개수 1,   , 채널 1
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
