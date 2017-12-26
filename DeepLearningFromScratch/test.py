from mnist import load_mnist
import numpy as np
from twoLayerNet import TwoLayerNet
import matplotlib.pyplot as plt
from functions import *


input_data = np.random.randn(1000, 100)
node_num = 100 # 은칙층 노드 수
hidden_layer_size = 10

# 활성화값
activations = {}

x = input_data
for i in range(hidden_layer_size):

    # 이전 계층 출력을 입력으로 사용
    if i != 0:
        x = activations[i-1]

    # 가중치 초깃값을 다양하게 바꿔가며 실험해보자！
    w = np.random.randn(node_num, node_num) * 1
    #w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    #w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    # Affine
    a = np.dot(x, w)

    # 활성화 함수도 바꿔가며 실험해보자！
    z = sigmoid(a)
    #z = relu(a)
    #z = np.tanh(a)

    activations[i] = z


# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    if i != 0:
        plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()
