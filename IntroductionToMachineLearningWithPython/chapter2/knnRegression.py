# k-NN 회귀

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

x, t = mglearn.datasets.make_wave(n_samples=40)
x_train, x_test, t_train, t_test = train_test_split(x, t, random_state=0)

# 3개의 그래프 생성
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# -3과 3 사이에 균등한 간격의 점을 1000개의 만든 후, 2차원 행렬(1000행, 1열 - feature가 1개인 1000개의 데이터)로 변환
line = np.linspace(-3, 3, 1000).reshape(-1, 1)


for n_neighbors, axis in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors).fit(x_train, t_train)
    pred = reg.predict(line)
    axis.plot(line, pred) # -3 ~ 3 사이의 1000개의 점에 대한 예측선을 그린다.
    axis.plot(x_train, t_train, '^', c=mglearn.cm2(0), markersize=8) # 훈련 데이터에 대한 그래프
    axis.plot(x_test, t_test, 'v', c=mglearn.cm2(1), markersize=8) # 테스트 데이터에 대한 그래프

    axis.set_title("Neighbors {}'s Training Score: {:.2f} Test Score: {:.2f}".format(
        n_neighbors, reg.score(x_train, t_train), reg.score(x_test, t_test))) # 훈련 데이터와 테스트 데이터에 대한 정확도 표시

    axis.set_xlabel("Feature")
    axis.set_ylabel("Label")

axes[0].legend(["Model Prediction", "Training Data/Label", "Test Data/Label"], loc="best")

plt.show()

