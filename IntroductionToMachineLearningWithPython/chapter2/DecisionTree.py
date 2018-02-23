# k-NN 분류

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.tree import *
import graphviz


# scikit-learn의 데이터셋 구조
#  - feature_names: 각 훈련데 데이터 속성들의 이름
#  - data: 샘플(데이터 포인트) - 훈련 데이터, 하나의 샘플은 feature_names의 수 만큼의 속성을 갖는다.
#  - target_names: 정답 레이블의 이름
#  - target: 레이블(정답)
#  - DESCR: 설명


# scikit-learn의 유방암 데이터 셋 load
cancer = load_breast_cancer()

# 데이터를 랜덤으로 섞은 후, 훈련데이터(x_train, x_test)와 테스트 데이터(t_train, t_test) 분리 - 75:25
x_train, x_test, t_train, t_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(x_train, t_train)
print("훈련 셋 정확도: {:.3f}".format(tree.score(x_train, t_train)))
print("훈련 셋 정확도: {:.3f}".format(tree.score(x_test, t_test)))
print("특성 중요도:\n{}".format(tree.feature_importances_))
export_graphviz(tree, out_file="tree.dot", class_names=["Neg", "Pos"], feature_names=cancer.feature_names, impurity=False, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()


# 특성 중요도 그래프
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(tree)

plt.show()