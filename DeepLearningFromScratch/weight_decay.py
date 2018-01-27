from multiLayerNet import MultiLayerNet
from mnist import load_mnist
from optimizer import *
import numpy as np
import matplotlib.pyplot as plt



# 데이터 읽기
(train_data, train_label), (test_data, test_label) = load_mnist()

# 오버피팅을 재현하기 위해 학습 데이트를 줄임
train_data = train_data[:300]
train_label = train_label[:300]

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        optimizer=SGD(), use_batchnorm=False, weight_decay_lambda=0.1)


max_epochs = 201

train_size = train_data.shape[0]
batch_size = 100

train_acc_list = []
test_acc_list = []

iter_per_epoch = train_size / batch_size

epoch_count = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = train_data[batch_mask]
    t_batch = train_label[batch_mask]

    network.train(x_batch, t_batch)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(train_data, train_label)
        test_acc = network.accuracy(test_data, test_label)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch %03d: Train accuracy = %2.2f%%, Test accuracy = %2.2f%%" %
              (epoch_count, train_acc * 100.0, test_acc * 100.0))

        epoch_count += 1
        if epoch_count >= max_epochs:
            break


# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
