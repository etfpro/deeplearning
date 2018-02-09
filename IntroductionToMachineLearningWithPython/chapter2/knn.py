import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 26개의 (x1, x2)와 그에 대한 레이블(0 or 1)을 임으로 생성
x, t = mglearn.datasets.make_forge()
print(x.shape)

# 데이터를 랜덤으로 섞은 후, 훈련데이터(x_train, x_test)와 테스트 데이터(t_train, t_test) 분리 - 75:25
x_train, x_test, t_train, t_test = train_test_split(x, t, random_state=0)

"""
# k-NN 분류기 생성 후 모델 생성
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(x_train, t_train)

# 예측(추론) 및 정확도 계산
#prediction = clf.predict(x_test)
acc = clf.score(x_test, t_test)
print("정확도: {:.2f}".format(acc))
"""

fig, axes = plt.subplots(1, 5, figsize=(15, 6))
for n_neighbors, axis in zip([1, 3, 9, 12, 15], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(x, t)
    mglearn.plots.plot_2d_separator(clf, x, fill=True, eps=0.5, ax=axis, alpha=.4)
    mglearn.discrete_scatter(x[:, 0], x[:, 1], t, ax=axis)
    axis.set_title("{} neighbors".format(n_neighbors))
    axis.set_xlabel("Feature 0")
    axis.set_ylabel("Feature 1")
axes[0].legend(loc=3)

plt.show()

pass