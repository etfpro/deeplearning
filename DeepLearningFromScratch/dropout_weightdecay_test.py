from common.multiLayerNet import MultiLayerNet
from common.mnist import load_mnist
from common.optimizer import *
from common.trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt


# 데이터 읽기
(train_data, train_label), (test_data, test_label) = load_mnist()

# 오버피팅을 재현하기 위해 학습 데이트를 줄임
train_data = train_data[:300]
train_label = train_label[:300]

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        optimizer=Adam(), use_batchnorm=True, weight_decay_lambda=0, dropout_ratio=-0)

trainer = Trainer(network, train_data, train_label, test_data, test_label, epochs=201, verbose=True)
trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list


# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()



