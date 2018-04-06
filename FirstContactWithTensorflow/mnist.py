import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/", one_hot=True)

print(tf.convert_to_tensor(mnist.train.images).get_shape())

num_input = 784
num_class = 10

W = tf.Variable(tf.random_normal([num_input, num_class], stddev=0.01))
b = tf.Variable(tf.zeros(num_class))

X = tf.placeholder(tf.float32, [None, num_input]) # 1개의 이미지는 28 X 28 = 784 픽셀
t = tf.placeholder(tf.float32, [None, num_class]) # 1개의 이미지에 대한 label은 10가지 클래스 중 하나 : One-hot encoding

y = tf.nn.softmax(tf.matmul(X, W) + 10)

error = -tf.reduce_sum(t * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(error)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    batch_x, batch_t = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={X: batch_x, t:batch_t})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={X: mnist.test.images, t: mnist.test.labels}))
sess.close()
