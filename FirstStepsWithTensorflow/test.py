import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def testRegression():

    vectors_set = []

    for i in range(1000):
        x = np.random.normal(0, 0.55)
        y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
        vectors_set.append([x, y])


    x_data = [v[0] for v in vectors_set] # 입력 리스트
    y_data = [v[1] for v in vectors_set] # 정답 리스트

    """
    
    plt.plot(x_data, y_data, 'bo')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    """


    # 가중치 초기값 -1 ~ 1 사이 랜덤값(균등분포) - 1차원 vector 변수 텐서 선언
    W = tf.Variable(tf.random_uniform([1], -1, 1))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b

    # 오차함수(평균제곱함수)
    loss = tf.reduce_mean(tf.square(y - y_data))

    # 경사하강
    optimizer = tf.train.GradientDescentOptimizer(0.5)

    # 경사하강법으로 훈련 객체 생성
    train = optimizer.minimize(loss)

    # 위에 선언된 변수들 초기화
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    # 모든 데이터를 20번 훈련?
    for step in range(20):
        sess.run(train)
        print(step, sess.run(loss), sess.run(W), sess.run(b))



    plt.plot(x_data, y_data, 'bo')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()



def testKMeans():
    num_points = 2000
    vectors_set = []
    for i in range(num_points):
        if np.random.normal() > 0.5:
            vectors_set.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
        else:
            vectors_set.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

    df = pd.DataFrame({"x": [v[0] for v in vectors_set],
                       "y": [v[1] for v in vectors_set]})
    sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
    plt.show()

    vectors = tf.constant(vectors_set)
    k = 4

    # 첫번째 차원을 기준으로 텐서의 원소를 섞은 후, 텐서의 일부분 삭제하여 중심점을 구한다.
    centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [k,-1]))

    expanded_vectors = tf.expand_dims(vectors, 0)
    expanded_centroides = tf.expand_dims(centroides, 1)

testKMeans()