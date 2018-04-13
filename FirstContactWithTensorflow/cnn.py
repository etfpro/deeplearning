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
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Conv 연산
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# Max Pool 연산
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Conv 계층 생성(폭, 높이, 채널, 필터 수) 및 연산
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_images, W_conv1) + b_conv1) # 활성화 함수
h_pool1 = max_pool(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 활성화 함수
h_pool2 = max_pool(h_conv2)


# FC 계층 - Hidden Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# FC 가중합 & 활성화(Relu) 연산
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) # 4차원 -> 2차원
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax - Output Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


################################################################################

# 오차함수: CEE
cee = -tf.reduce_sum(t * tf.log(y))

# 훈련
train = tf.train.AdadeltaOptimizer(1e-4).minimize(cee)

# 정확도 측정
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    x_batch = mnist.train.next_batch(100)
    if i % 100 == 0: # 100번 마다 훈련 정확도 측정
        train_accuracy = sess.run(accuracy, feed_dict={X:x_batch[0], t:x_batch[1], keep_prob:1.0})
        print("Step %d, Training accuracy %g" % (i, train_accuracy))

    sess.run(train, feed_dict={X:x_batch[0], t:x_batch[1], keep_prob:0.5})

print(">> Test accuracy %g" % sess.run(accuracy, feed_dict={X:mnist.test.images, t:mnist.test.labels, keep_prob:1.0}))

sess.close();