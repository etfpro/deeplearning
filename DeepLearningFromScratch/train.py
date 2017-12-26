from itertools import starmap

import numpy as np
from mnist import load_mnist
from twoLayerNet import TwoLayerNet
import time
from optimizer import *


# 데이터 읽기
(train_data, train_label), (test_data, test_label) = load_mnist(one_hot_label=True)

# 각 배치 학습별 손실값을 저장하는 리스트
train_loss_list = []

# 훈련데이터에 대한 정확도를 저장하는 리스트
train_acc_list = []

# 테스트데이터에 대한 정확도를 저장하는 리스트
test_acc_list = []


# 하이퍼파라메터
iters_num = 10000 # SGD 반복회수
train_size = train_data.shape[0] # 훈련데이터 개수
batch_size = 100 # 미니배치 크기

# 1 주기 당 SGD 반복 회수: 전체 훈련데이터를 1회 학습하기 위한 SGD 반복 회수
iter_per_epoch = int(train_size / batch_size)

# 신경망 생성
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
learningRate = 0.01
#network.optimizer = SGD(learningRate)
#network.optimizer = Momentum(learningRate)
#network.optimizer = AdaGrad(learningRate)
#network.optimizer = RMSprop(learningRate)
network.optimizer = Adam(learningRate)



for i in range(iters_num):

    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = train_data[batch_mask]
    t_batch = train_label[batch_mask]

    # 기울기 계산 및 가중치 갱신
    #network.train(x_batch, t_batch)
    grad = network.numerical_gradient(x_batch, t_batch)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learningRate * grad[key]

    # 학습 경과 기록: 매 SGD 당 손실값 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    print(">> Loss", i, ": ", loss)
    # 1 주기 당 정확도 계산 & 기록
    if i % iter_per_epoch == 0:
        # 훈련데이터에 대한 정확도 계산 & 기록
        train_acc = network.accuracy(train_data, train_label)
        train_acc_list.append(train_acc)

        # 테스트데이터에 대한 정확도 계산 & 기록
        test_acc = network.accuracy(test_data, test_label)
        test_acc_list.append(test_acc)

        print("Train accuracy = %.2f%%, Test accuracy = %.2f%%" % (train_acc * 100.0, test_acc * 100.0))





