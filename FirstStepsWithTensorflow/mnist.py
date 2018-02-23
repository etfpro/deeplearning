import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 데이터 다운로드
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)


################################################################################
# 입력, 레이블 데이터를 위한 placeholder
################################################################################
X = tf.placeholder(tf.float32, [None, 784]) # 1개의 이미지는 28 X 28 = 784 픽셀
t = tf.placeholder(tf.float32, [None, 10]) # 1개의 이미지에 대한 label은 10가지 클래스 중 하나 : One-hot encoding


################################################################################
# 신경망 구성(3계층)
################################################################################
# dropout 시 노드 유지 비율
keep_prob = tf.placeholder(tf.float32)

# 은닉층 1
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01)) # 표준편차 0.01
b1 = tf.Variable(tf.zeros([256]))
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L1 = tf.nn.dropout(L1, keep_prob) # dropout 시 노드 유지 비율

# 은닉층 2
W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01)) # 표준편차 0.01
b2 = tf.Variable(tf.zeros([256]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
L2 = tf.nn.dropout(L2, 0.8) # dropout - 80% 노드만 사용

# 출력층
W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01)) # 표준편차 0.01
b3 = tf.Variable(tf.zeros([10]))
model = tf.add(tf.matmul(L2, W3), b3)


################################################################################
# 손실함수 및 훈련 옵티마이저
################################################################################
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=t))
global_step = tf.Variable(0, trainable=False, name='global_step')  # 학습 횟수를 카운트하는 변수
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost, global_step=global_step)


################################################################################
# 학습 진행
################################################################################

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

        # train과 cost 두개의 연산을 동시에 수행
        _, cost_val = sess.run([train_op, cost], feed_dict={X:batch_x, t:batch_t, keep_prob:0.8})
        #print("Step {}: cost - {}".format(sess.run(global_step), cost_val))

        total_cost += cost_val

    # 각 주기마다 평균 손실값 출력
    print("Epoch({}): Average Cost = {:.3f}".format(epoch + 1, total_cost / total_batch))


################################################################################
# 결과 확인
################################################################################
# 예측
prediction = sess.run(model, feed_dict={X: mnist.test.images, t: mnist.test.labels, keep_prob: 1})
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


################################################################################
# 결과 확인
################################################################################

