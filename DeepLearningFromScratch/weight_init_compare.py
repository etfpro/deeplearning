# coding: utf-8
from common.mnist import load_mnist
from common.util import smooth_curve
from common.multiLayerNet import MultiLayerNet
from common.optimizer import *
import matplotlib.pyplot as plt


# 0. MNIST 데이터 읽기==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# 1. 실험용 설정==========
weight_init_types = {'std(0.01)': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
markers = {'std(0.01)': 'o', 'Xavier': 's', 'He': 'D'}
optimizer = SGD()
#optimizer = Momentum()
#optimizer = AdaGrad()
#optimizer = Adam()

networks = {}
train_loss = {}
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                                  output_size=10, weight_init_std=weight_type, optimizer=optimizer, use_batchnorm=True)
    train_loss[key] = []


# 2. 훈련 시작==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]


    # 옵티마이저 별로 미니배치에 대한 가중치 갱신
    for key in weight_init_types:
        network = networks[key]
        network.train(x_batch, t_batch)
        train_loss[key].append(network.lossValue) # 학습 경과 기록: 매 SGD 당 손실값 기록

    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types:
            print(key + ":" + str(networks[key].loss(x_batch, t_batch, True)))


# 3. 그래프 그리기==========
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()
