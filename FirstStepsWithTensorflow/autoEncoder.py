import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


################################################################################
# 하이퍼파라메터 & 입력
################################################################################
learning_rate = 0.01
epochs = 20
batch_size = 100
n_hidden = 256 # 은닉노드 수(색상의 수와 동일)
n_input = 28 * 28 # 입력노드 수


# 입력 데이터를 위한 placeholder
X = tf.placeholder(tf.float32, [None, n_input]) # 28 X 28 이미지


################################################################################
# 신경망: 인코더 & 디코드
################################################################################
# 인코더
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))

encoder = tf.add(tf.matmul(X, W_encode), b_encode) # 가중합
encoder = tf.nn.sigmoid(encoder) # 활성화함수


# 디코더(출력층)
W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))

decoder = tf.add(tf.matmul(encoder, W_decode), b_decode) # 가중합
decoder = tf.nn.sigmoid(decoder) # 활성화함수


################################################################################
# 학습용 연산
################################################################################

# 손실함수: 입력값과 출력값의 차이에 대한 평균제곱오차 함수
cost = tf.reduce_mean(tf.pow(X - decoder, 2))

# 옵티마이저
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)



################################################################################
# 학습 진행
################################################################################

# MNIST 데이터 다운로드
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)


# 세션 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 미니배치 사이즈
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(epochs):
    total_cost = 0 # 주기 당 손실값
    for i in range(total_batch):
        batch_x, _ = mnist.train.next_batch(batch_size)

        # train과 cost 두개의 연산을 동시에 수행
        _, cost_val = sess.run([optimizer, cost], feed_dict={X:batch_x})
        #print("Step {}: cost - {}".format(sess.run(global_step), cost_val))

        total_cost += cost_val

    # 각 주기마다 평균 손실값 출력
    print("Epoch({}): Average Cost = {:.3f}".format(epoch + 1, total_cost / total_batch))



################################################################################
# 결과 확인
################################################################################
# 예측
sample_size = 10
prediction = sess.run(decoder, feed_dict={X: mnist.test.images[:sample_size]})

fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))
for i in  range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()

    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    ax[1][i].imshow(np.reshape(prediction[i], (28, 28)))

plt.show()

sess.close()
