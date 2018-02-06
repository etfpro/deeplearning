import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()

x_train, x_test, t_train, t_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

"""
iris_dataframe = pd.DataFrame(x_train, columns=iris_dataset.feature_names)

# 데이터프레임을 사용해 t_train(정답 - iris 종류)에 따라 색으로 구분된 산점도 행렬을 만든다
pd.plotting.scatter_matrix(iris_dataframe, c=t_train, figsize=(15, 15), marker='o', hist_kwds={"bins": 20}, s=60, alpha=0.8, cmap=mglearn.cm3)

plt.show()
"""

# k-NN 분류기
knn = KNeighborsClassifier(n_neighbors=1)

# 훈련 데이터셋으로 모델을 만듬(학습)
knn.fit(x_train, t_train)


# 예측하기
x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(x_new) # 리턴되는 예측값은 fit()의 두번째 인자로 전달된 Label 리스트(t_train)에서 정답에 해당하는 값
print("예측: {}".format(prediction))
print("예측한 클래스: {}".format(iris_dataset['target_names'][prediction]))

# 테스트 데이터를 이용한 정확도 측정
y_pred = knn.predict(x_test)
accuracy = np.mean(y_pred == t_test)
# or
accuracy = knn.score(x_test, t_test)
print("Accuracy = {:.3f}".format(accuracy))




