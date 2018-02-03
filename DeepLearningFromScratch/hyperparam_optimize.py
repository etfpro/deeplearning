import numpy as np
import matplotlib.pyplot as plt
from common.mnist import load_mnist
from common.multiLayerNet import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer
from common.optimizer import *

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 결과를 빠르게 얻기 위해 훈련 데이터를 줄임
x_train = x_train[:500]
t_train = t_train[:500]

# 20%를 검증 데이터로 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate) # 검증 데이터 수

# 원래의 훈련 데이터(500개)를 섞는다.
x_train, t_train = shuffle_dataset(x_train, t_train)

# 원래의 훈련 데이터 500개에서 20%인 100개를 검증 데이터로 사용
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]

# 검증 데이터를 제외한 나머지 400개를 훈련 데이터로 사용
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]



# 400개의 훈련 데이터로 훈련하고, 100개의 검증 데이터로 테스트
def __train(lr, weight_decay, epocs=10):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                            optimizer=SGD(lr=lr), use_batchnorm=False, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, batch_size=100, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list



# 하이퍼파라미터(학습률, 가중치감소율) 무작위 탐색======================================
optimization_trial = 100

results_val = {} # 검증 데이터에 대한 정확도 리스트 저장 딕셔너리
results_train = {} # 훈련 데이터에 대한 정확도 리스트 저장 딕셔너리

for i in range(optimization_trial):
    # 탐색한 하이퍼파라미터의 범위 지정===============
    # 가중치감소율 초기화 (10^-8 ~ 10^-1 사이의 균등분포를 가진 난수)
    weight_decay = 10 ** np.random.uniform(-8, -4)

    # 학습률 초기화 (10^-6 ~ 10^-1 사이의 균등분포를 가진 난수)
    lr = 10 ** np.random.uniform(-6, -2)


    # ================================================
    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print("Trial " + str(i) +
          " | Validation Accuracy:" + str(np.round(val_acc_list[-1] * 100, 2)) + "%" # 검증 데이터 정확도(마지막 정확도) 
          " | Learning Rate:" + str(np.round(lr, 6)) + # 학습률
          ", Weight Decay:" + str(np.round(weight_decay, 7))) # 가중치감소율
    key = "learning rate:" + str(np.round(lr, 6)) + ", weight decay:" + str(np.round(weight_decay, 7)) # 가중치감소율
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list



# 그래프 그리기========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 10
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

# 검증 데이터의 정확도를 기준으로 정렬
# item[0] - key, item[1] = value
for key, val_acc_list in sorted(results_val.items(), key=lambda item: item[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(Validation Accuracy:" + str(np.round(val_acc_list[-1] * 100, 3)) + "%) | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: #
        plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.xlabel("trials")
    plt.ylabel("Accuracy")
    plt.plot(x, val_acc_list) # 검증 데이터 정확도 리스트
    plt.plot(x, results_train[key], "--") # 훈련 데이터 정확도 리스트
    i += 1

    if i >= graph_draw_num:
        break

plt.show()
