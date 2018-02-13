# 선형 분류

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.svm import *


# scikit-learn의 데이터셋 구조
#  - feature_names: 각 훈련데 데이터 속성들의 이름
#  - data: 샘플(데이터 포인트) - 훈련 데이터, 하나의 샘플은 feature_names의 수 만큼의 속성을 갖는다.
#  - target_names: 정답 레이블의 이름
#  - target: 레이블(정답)
#  - DESCR: 설명


# 유방암 데이터셋 로드
cancer = load_breast_cancer()

# 데이터를 랜덤으로 섞은 후, 훈련데이터(x_train, x_test)와 테스트 데이터(t_train, t_test) 분리 - 75:25
x_train, x_test, t_train, t_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# 로지스틱회귀(분류)
logreg = LogisticRegression(C=100).fit(x_train, t_train)
print("훈련셋 정확도: {:.3f}".format(logreg.score(x_train, t_train)))
print("테스트셋 정확도: {:.3f}".format(logreg.score(x_test, t_test)))




#
x, t = make_blobs(random_state=42)
linear_svm = LinearSVC().fit(x, t)
print("계수 배열의 크기: ", linear_svm.coef_.shape)
print("절편 배열의 크기: ", linear_svm.intercept_.shape)


mglearn.discrete_scatter(x[:, 0], x[:, 1], t)

# -15 ~ 15 사이의 50개의 숫자 리턴(x 좌표)
line = np.linspace(-15, 15)
for w, b, color in zip(linear_svm.coef_, linear_svm.intercept_, mglearn.cm3.colors):
    print("w = {}".format(w))
    plt.plot(line,  -(line * w[0] + b) / w[1], c=color) # 분류선을 속성 1의

plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2", "Class 0 border", "Class 1 border", "Class 2 border"])
plt.show()

