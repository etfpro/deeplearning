from itertools import starmap

import numpy as np
from mnist import load_mnist
from twoLayerNet import TwoLayerNet
import time
from optimizer import *


# 데이터 읽기
(train_data, train_label), (test_data, test_label) = load_mnist(one_hot_label=True)

# 각 배치 학습별 손실값 변화 히스토리 저장
train_loss_list = []

# 훈련 데이터에 대한 정확도 변화 히스토리 저장
train_acc_list = []

# 테스트 데이터에 대한 정확도 변화 히스토리 저장
test_acc_list = []


# 하이퍼파라메터
iters_num = 1000 # SGD 반복회수
train_size = train_data.shape[0] # 훈련데이터 개수
batch_size = 100 # 미니배치 크기
learning_rate = 0.3 # 학습률

# 1 주기 당 SGD 반복 회수: 전체 훈련데이터를 1회 학습하기 위한 SGD 반복 회수
iter_per_epoch = int(train_size / batch_size)

# 신경망 생성
optimizer = SGD(learning_rate)
#optimizer = Momentum(learning_rate)
#optimizer = AdaGrad(learning_rate)
#optimizer = RMSprop(learning_rate)
#network.optimizer = Adam(learning_rate)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, optimizer=optimizer)

epoch_index = 0
for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = train_data[batch_mask]
    t_batch = train_label[batch_mask]

    # 미니배치에 대한 가중치 갱신
    network.train(x_batch, t_batch)

    # 학습 경과 기록: 매 SGD 당 손실값 기록
    train_loss_list.append(network.lossValue)

    # 1 주기 당 정확도 계산 & 기록
    if i % iter_per_epoch == 0:
        # 전체 훈련데이터에 대한 정확도 계산 & 기록
        train_acc = network.accuracy(train_data, train_label)
        train_acc_list.append(train_acc)

        # 전체 테스트 데이터에 대한 정확도 계산 & 기록
        test_acc = network.accuracy(test_data, test_label)
        test_acc_list.append(test_acc)

        print("Epoch %d: Train accuracy = %.2f%%, Test accuracy = %.2f%%" %
              (epoch_index, train_acc * 100.0, test_acc * 100.0))

        epoch_index += 1



#markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()
