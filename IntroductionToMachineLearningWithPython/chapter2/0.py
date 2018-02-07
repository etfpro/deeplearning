
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


# scikit-learn의 유방암 데이터 셋
cancer = load_breast_cancer()
print("클래스별 샘플 개수:\n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))


# scikit-learn의 보스턴 주택가격 데이터 셋
boston = mglearn.datasets.load_extended_boston()
print(type(boston))
