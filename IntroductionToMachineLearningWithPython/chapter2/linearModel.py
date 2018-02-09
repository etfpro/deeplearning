# 선형 모델

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *

# 보스톤 주택가격 데이터셋
# 506개의 샘플, 샘플당 105개의 특성
x, t = mglearn.datasets.load_extended_boston()

# 데이터를 랜덤으로 섞은 후, 훈련데이터(x_train, x_test)와 테스트 데이터(t_train, t_test) 분리 - 75:25
x_train, x_test, t_train, t_test = train_test_split(x, t, random_state=0)
print("t(lable) of train:\n{}".format(t_train))

################################################################################
# 선형 회귀
################################################################################
r = LinearRegression().fit(x_train, t_train)
print(">> 선형 회귀 <<")
print("훈련 셋 정확도: {:.2f}".format(r.score(x_train, t_train)))
print("테스트 셋 정확도: {:.2f}".format(r.score(x_test, t_test)))



################################################################################
# 릿지(ridge) 회귀 - 가중치 감소 기법(L2)
################################################################################
r = Ridge(alpha=1).fit(x_train, t_train)
print(">> 릿지 회귀 <<")
print("훈련 셋 정확도: {:.2f}".format(r.score(x_train, t_train)))
print("테스트 셋 정확도: {:.2f}".format(r.score(x_test, t_test)))



################################################################################
# 라쏘(lasso) 회귀 - 가중치 감소 기법(L1)
################################################################################
r = Lasso(alpha=1).fit(x_train, t_train)
print(">> 라쏘 회귀 <<")
print("훈련 셋 정확도: {:.2f}".format(r.score(x_train, t_train)))
print("테스트 셋 정확도: {:.2f}".format(r.score(x_test, t_test)))
print("사용한 특성의 수: {}".format(np.sum(r.coef_ != 0)))
print(r.n_iter_)