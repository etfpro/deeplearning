from common.mnist import load_mnist
from common.multiLayerNet import MultiLayerNet
from common.optimizer import *
import matplotlib.pyplot as plt
from common.util import smooth_curve



# 데이터 읽기
(train_data, train_label), (test_data, test_label) = load_mnist()

# 하이퍼파라메터
iters_num = 2000 # SGD 반복회수
train_size = train_data.shape[0] # 훈련데이터 개수
batch_size = 128 # 미니배치 크기
learning_rate = 0.01 # 학습률

# 1 주기 당 SGD 반복 회수: 전체 훈련데이터를 1회 학습하기 위한 SGD 반복 회수
iter_per_epoch = int(train_size / batch_size)


optimizers = {"SGD":SGD(learning_rate), "Momentum":Momentum(learning_rate), "AdaGrad":AdaGrad(learning_rate), "RMSProp":RMSProp(learning_rate), "Adam":Adam(learning_rate)}
markers = {'SGD': 'o', 'Momentum': 's', 'AdaGrad': 'd', 'RMSProp':'.', 'Adam':'x'}
colors = {'SGD': 'red', 'Momentum': 'blue', 'AdaGrad': 'black', 'RMSProp':'magenta', 'Adam':'green'}

# 신경망 생성
networks = {}
train_loss = {} # 각 배치 학습별 손실값 변화 히스토리 저장
for key in optimizers:
    #networks[key] = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, optimizer=optimizers[key])
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10, optimizer=optimizers[key])
    train_loss[key] = []


epoch_index = -1

# 훈련
for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = train_data[batch_mask]
    t_batch = train_label[batch_mask]

    if i % iter_per_epoch == 0:
        epoch_index += 1
        print(">> Epoch %d <<" % epoch_index)

    # 옵티마이저 별로 미니배치에 대한 가중치 갱신
    for key in optimizers:
        network = networks[key]
        network.train(x_batch, t_batch)
        train_loss[key].append(network.lossValue) # 학습 경과 기록: 매 SGD 당 손실값 기록

        # 1 주기 당 전체 훈련/테스트 데이터에 대한 정확도 계산
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(train_data, train_label)
            test_acc = network.accuracy(test_data, test_label)
            print("%10s: Train accuracy = %2.2f%%, Test accuracy = %2.2f%%" %
                (key, train_acc * 100.0, test_acc * 100.0))



# 손실값에 대한 그래프 그리기
x = np.arange(iters_num)
for key in optimizers:
    plt.plot(x, smooth_curve(train_loss[key]), label=key, color=colors[key], linewidth=0.7, marker=markers[key], markevery=100)

plt.title("MNIST Train")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.ylim(0, 1.0)
plt.legend()
plt.show()

