# k-NN 분류

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



# scikit-learn의 데이터셋 구조
#  - feature_names: 각 훈련데 데이터 속성들의 이름
#  - data: 샘플(데이터 포인트) - 훈련 데이터, 하나의 샘플은 feature_names의 수 만큼의 속성을 갖는다.
#  - target_names: 정답 레이블의 이름
#  - target: 레이블(정답)
#  - DESCR: 설명


# scikit-learn의 유방암 데이터 셋 load
cancer = load_breast_cancer()

# 데이터를 랜덤으로 섞은 후, 훈련데이터(x_train, x_test)와 테스트 데이터(t_train, t_test) 분리 - 75:25
x_train, x_test, t_train, t_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)


training_accuracy = [] # 각 모델별 훈련 데이터에 대한 정확도 저장 리스트
test_accuracy = [] # 각 모델별 테스트 데이터에 대한 정확도 저장 리스트

# 1에서 10까지 최근접 이웃을 조정하면서 모델 생성 후 정확도 측정
neighbor_counts = range(1, 11)
for n_neighbor in neighbor_counts:
    # 모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbor)
    clf.fit(x_train, t_train)

    # 각 모델별 훈련 데이터에 대한 정확도 측정 후 저장
    training_accuracy.append(clf.score(x_train, t_train))

    # 각 모델별 테스트 데이터에 대한 정확도 측정 후 저장
    test_accuracy.append(clf.score(x_test, t_test))


plt.plot(neighbor_counts, training_accuracy, label="Trainning Accuracy")
plt.plot(neighbor_counts, test_accuracy, label="Test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()
pass

