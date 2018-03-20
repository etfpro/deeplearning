import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


num_points = 1000

vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55) # 평균 0, 표준편차 0.55인 정규분포에서 랜덤값을 얻는다.
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03) # 직선을 약간 정규분포로 분포시킨다.
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set] # 트레이닝 데이터
y_data = [v[1] for v in vectors_set] # 레이블


W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # -1.0 ~ 1.0 사이의 값으로 초기화
b = tf.Variable(tf.zeros([1])) # 0으로 초기화

# 모델
y = W * x_data + b

# 손실함수
loss = tf.reduce_mean(tf.square(y - y_data)) # 평균(차원축소)

# 가중치 업데이트(훈련)
optimizer = tf.train.GradientDescentOptimizer(0.1) # 경사하강법
train = optimizer.minimize(loss)

# 실행
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 변수 초기화

for i in range(200):
    sess.run(train)
    print(i, sess.run(W), sess.run(b))

plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

sess.close()