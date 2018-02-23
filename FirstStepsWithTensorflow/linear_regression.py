import tensorflow as tf


# 가중치 및 편향 최기화: 학습 대상
# 1개의 값을 랜덤값으로 초기화
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 입력과 정답
# placeholder는 매개변수(추후 재 활용)
X = tf.placeholder(tf.float32, name="X")
t = tf.placeholder(tf.float32, name="t")

# 추론모델(선형함수)
h = W * X + b

# 비용함수 정의(평균제곱오차, MSE)
# reduce_mean: 차원 축소 평균 계산
cost = tf.reduce_mean(tf.square(h - t))

# 학습 옵티마이저
optipmizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# 학습 연산: 비용함수를 최소화
train_op = optipmizer.minimize(cost)

# 실제 학습 수행
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 100번 학습
    for step in range(150):
        # train와 cost 두개의 연산을 동시에 수행
        _, cost_val = sess.run([train_op, cost], feed_dict={X: [1, 2, 3], t: [1, 2, 3]})
        print(step, cost_val, sess.run(W), sess.run(b))


    # 테스트
    print("\n=== Test ===")
    print("X: 5, Y:", sess.run(h, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(h, feed_dict={X: 2.5}))

