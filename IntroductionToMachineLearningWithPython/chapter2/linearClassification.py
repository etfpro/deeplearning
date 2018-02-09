# 선형 분류

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.svm import *

# 26개의 (x1, x2)와 그에 대한 레이블(0 or 1)을 임으로 생성
x, t = mglearn.datasets.make_forge()

_, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    classifier = model.fit(x, t)
    mglearn.plots.plot_2d_separator(classifier, x, fill=False, eps=0.5, ax=ax, alpha=.7)
    mglearn.discrete_scatter(x[:, 0], x[:, 1], t, ax=ax)
    ax.set_title(classifier.__class__.__name__)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()

plt.show()
