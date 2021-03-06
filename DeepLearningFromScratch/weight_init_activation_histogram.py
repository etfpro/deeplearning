# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import common.functions as func


input_data = np.random.randn(1000, 100)  # 1000개의 데이터(평균 0, 표준편차 1)
node_num = input_data.shape[1]  # 입력층의 노드 수, 각 은닉층의 노드 수로 사용
hidden_layer_size = 15
activations = {}  # 이곳에 활성화 결과를 저장

x = input_data

for i in range(hidden_layer_size):
    # 초깃값을 다양하게 바꿔가며 실험해보자！
    #w = np.random.randn(node_num, node_num) * 0.01
    #w = np.random.randn(node_num, node_num) * 0.01
    #w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num) # Xavier 초기값
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num) # He 초기값(ReLU 용)

    if i != 0:
        x = activations[i-1] # 이전 계층의 출력값(활성화값)
    a = np.dot(x, w)

    # 활성화 함수도 바꿔가며 실험해보자！
    #z = func.sigmoid(a)
    #z = np.tanh(a)
    z = func.relu(a)

    activations[i] = z # 활성화값 저장 (1,000 X 100 개)


# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0:
        plt.yticks([], [])
    #plt.xlim(0, 2)
    a = a.flatten()
    plt.hist(a, 30, range=(0.0, max(1.0, np.max(a))))
    #plt.hist(a, 30, range=(0, 1))
plt.show()
