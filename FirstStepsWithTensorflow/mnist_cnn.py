import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from tensorflow.examples.tutorials.mnist import input_data



################################################################################
# 입력, 레이블 데이터를 위한 placeholder
################################################################################
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # 28 X 28 이미지, 색상 채널 1
t = tf.placeholder(tf.float32, [None, 10]) # 1개의 이미지에 대한 label은 10가지 클래스 중 하나 : One-hot encoding

# dropout 시 노드 유지 비율
keep_prob = tf.placeholder(tf.float32)

# 학습 중인지를 나타내는 변수
is_trining = tf.placeholder(tf.bool)


################################################################################
# 신경망 구성(3계층)
################################################################################
"""
# Conv계층 1: 32개의 필터
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) # 표준편차 0.01
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') # padding='SAME'옵션은 Conv계츧의 출력 크기를
                                                               # 입력 크기와 동일하게 padding 값을 설정하는 옵션
L1 = tf.nn.relu(L1)

# 풀링계층 1: 2 X 2 커널, 풀링계층의 stride는 풀링 커널의 크기와 동일하게 하는 것이 일반적
#           2 X 2 크기, stride 2인 풀링 계층은 입력의 크기(28 X 28)를 절반(14 X 14)으로 줄인다.
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Conv계층 2: 32개의 채널, 64개의 필터
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # 표준편차 0.01
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME') # padding='SAME'옵션은 Conv계츧의 출력 크기를
                                                                # 입력 크기와 동일하게 padding 값을 설정하는 옵션
L2 = tf.nn.relu(L2)

# 풀링계층 2: 입력의 크기(14 X 14)를 절반(7 X 7)로 줄인다.
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 완전연결 계층
W3 = tf.Variable(tf.random_normal([7 * 7 * 64,  256], stddev=0.01)) # 표준편차 0.01
L3 = tf.matmul(tf.reshape(L2, [-1, 7 * 7 * 64]), W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)


# 출력층
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01)) # 표준편차 0.01
model = tf.matmul(L3, W4)
"""

L1 = tf.layers.conv2d(X, 32, [3, 3])
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2]) # 2 X 2 풀링에 2 X 2 stride
L1 = tf.layers.dropout(L1, 0.7, is_trining)

L2 = tf.layers.conv2d(L1, 64, [3, 3])
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2]) # 2 X 2 풀링에 2 X 2 stride
L2 = tf.layers.dropout(L2, 0.7, is_trining)

L3 = tf.contrib.layers.flatten(L2)
L3 = tf.layers.dense(L3, 256, activation=tf.nn.relu)
L3 = tf.layers.dropout(L3, 0.5, is_trining)

model = tf.layers.dense(L3, 10, activation=None)


################################################################################
# 손실함수 및 훈련 옵티마이저
################################################################################
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=t))
global_step = tf.Variable(0, trainable=False, name='global_step')  # 학습 횟수를 카운트하는 변수
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost, global_step=global_step)


################################################################################
# 학습 진행
################################################################################

# MNIST 데이터 다운로드
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)


# 세션 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 미니배치 사이즈
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

epochs = 15
for epoch in range(epochs):
    total_cost = 0 # 주기 당 손실값
    for i in range(total_batch):
        batch_x, batch_t = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape(-1, 28, 28, 1) # 입력 데이터의 형상을 (데이터 수, 높이, 폭, 채널 수)로 변경

        # train과 cost 두개의 연산을 동시에 수행
        _, cost_val = sess.run([train_op, cost], feed_dict={X:batch_x, t:batch_t, keep_prob:0.7})
        #print("Step {}: cost - {}".format(sess.run(global_step), cost_val))

        total_cost += cost_val

    # 각 주기마다 평균 손실값 출력
    print("Epoch({}): Average Cost = {:.3f}".format(epoch + 1, total_cost / total_batch))


################################################################################
# 결과 확인
################################################################################
# 예측
prediction = sess.run(model, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                        t: mnist.test.labels, keep_prob: 1})
target = sess.run(t, feed_dict={t: mnist.test.labels})

# 정확도 측정
prediction = tf.argmax(prediction, axis=1)
target = tf.argmax(target, axis=1)

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) # true, false를 1, 0으로 변환 후, 평균을 낸다.
acc_value = sess.run(accuracy)

print("정확도: {:.2f} %".format(acc_value * 100))
print("Global Step:", sess.run(global_step))

sess.close()

