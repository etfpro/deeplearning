import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


################################################################################
# 2000개의 임의의 점(2차원 좌표) 생성
################################################################################
num_points = 2000
vector_points = []

for i in range(num_points):
    if np.random.random() > 0.5:
        vector_points.append([np.random.normal(0.0, 0.9),
                              np.random.normal(0.0, 0.9)])
    else:
        vector_points.append([np.random.normal(3.0, 0.5),
                              np.random.normal(1.0, 0.5)])

#df = pd.DataFrame({"x": [v[0] for v in vector_points],
#                   "y": [v[1] for v in vector_points]})
#sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
#plt.show()


################################################################################
# 0 단계: K개의 중심의 초기값 결정(랜덤)
################################################################################

# 분류할 점들(2000개의 랜덤한 점)을 상수 텐서로 변환
points = tf.constant(vector_points)

# k개의 중심 초기값을 무작위로 설정 - 분류할 2000개의 데이터를 무작위로 섞은 다음, k개의 중심점을 추출
k = 4
centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [k, -1]))


################################################################################
# 1 단계: 각 점들을 가장 가까운 군집(중심)에 할당
################################################################################

# 2차원 좌표인 k(4)개의 중심에 대한 모든 점들(2000개)에 대해 거리(3차원 - (4, 2000, 2))를 계산하기 위해서,
# 중심점과 점들을 3차원으로 확장 -> 확장된 차원은 크기가 1이기 때문에 브로드캐스팅 연산 가능
expanded_vectors = tf.expand_dims(points, 0) # (1, 2000, 2)
expanded_centroids = tf.expand_dims(centroids, 1) # (4, 1, 2)

# 분류할 각 점들과 중심점 사이의 거리 계산(유클리드 제곱 거리 - L2 distance) - (4, 20000)
diff = tf.subtract(expanded_vectors, expanded_centroids) # (중심, 데이터, 좌표) -> (4, 2000, 2)
square = tf.square(diff) # (4, 2000, 2)
distances = tf.reduce_sum(square, 2) # 좌표에 해당하는 마지막 차원(2)을 더한다. (4, 2000) -> 4개의 중심점과 2000개의 점사이의 거리

# 각 점들에 대한 중심과의 거리 중 작은 중심의 인덱스(군집번호)들만 추출(각 점에 대한 군집 할당)
assignments = tf.argmin(distances, 0) # (2000)


################################################################################
# 3 단계: 각 군집에 대해 새로운 중심 계산
#  - 각 군집에 속한 점들의 평균을 계산해서 새로운 중심으로 변경
################################################################################

# 4개의 군집에 대해서 각 군집에 속한 점들의 인덱스를 추출
# - tf.where(tf.equal(assignments, c)): 각 군집별로 군집에 속한 점들의 인덱스를 추출 - (n, 1)
# - tf.reshape(where, [1, -1]): (n, 1) 형태를 (1, n) 형태로 구조 변경
indices = [tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1]) for c in range(k)]

# 각점들을 각 군집으로(4, n, 2) 분류
gathers = [tf.gather(points, i) for i in indices]

# 각 군집들에 속해있는 점(x, y)들의 평균을 구한다.
means = [tf.reduce_mean(i, axis=[1]) for i in gathers] # (4, 1, 2)
means = tf.concat(axis=0, values=means) # (4, 2)

# 각 군집들의 점들의 평균값으로 중심을 변경
update_centroids = tf.assign(centroids, means)


################################################################################
# 4 단계: 1~3 단계 반복 실행
################################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

# 중심점 출력
print("centroids")
print(centroid_values)

# 각 점들을 DataFrame으로 변경
data = {"x": [], "y": [], "cluster": []}
for i in range(len(assignment_values)):
    data["x"].append(vector_points[i][0])
    data["y"].append(vector_points[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=7, hue="cluster", legend=False)
plt.show()

sess.close()
