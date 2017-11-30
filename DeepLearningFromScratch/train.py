from itertools import starmap

import numpy as np
from mnist import load_mnist
from twoLayerNet import TwoLayerNet
import time

# 데이터 읽기
(train_data, train_label), (test_data, test_label) = load_mnist(one_hot_label=True)

# 하이퍼파라메터
iters_num = 10000 # SGD 반복회수
train_size = train_data.shape[0] # 훈련데이터 개수
batch_size = 100 # 미니배치 크기
learning_rate = 0.1


network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 각 배치 학습별 손실값을 저장하는 리스트
train_loss_list = []

# 훈련데이터에 대한 정확도를 저장하는 리스트
train_acc_list = []

# 테스트데이터에 대한 정확도를 저장하는 리스트
test_acc_list = []


# 1 주기 당 SGD 반복 회수: 전체 훈련데이터를 1회 학습하기 위한 SGD 반복 회수
iter_per_epoch = int(train_size / batch_size)

start_time = time.time()
for i in range(iters_num):
    start_t = time.time()

    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = train_data[batch_mask]
    t_batch = train_label[batch_mask]

    # 기울기 계산
    network.train(x_batch, t_batch, learning_rate)

    # 학습 경과 기록: 매 SGD 당 손실값 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    print("실행[%d] - %f seconds. Loss = %f" % (i, time.time() - start_t, loss))

    # 1 주기 당 정확도 계산 & 기록
    if i % iter_per_epoch == 0:
        # 훈련데이터에 대한 정확도 계산 & 기록
        train_acc = network.accuracy(train_data, train_label)
        train_acc_list.append(train_acc)

        # 테스트데이터에 대한 정확도 계산 & 기록
        test_acc = network.accuracy(test_data, test_label)
        test_acc_list.append(test_acc)

        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))




end_time = time.time()

print(end_time - start_time)