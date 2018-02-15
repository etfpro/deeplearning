import pickle
import numpy as np
from common.layers import *
from common.optimizer import *
import matplotlib.pyplot as plt


# 다층 합성곱 신경망
# 정확도 99% 이상의 고정밀 합성곱 신경망
# 네트워크 구성은 아래와 같음
#    conv - relu - conv- relu - pool -
#    conv - relu - conv- relu - pool -
#    conv - relu - conv- relu - pool -
#    affine - relu - dropout - affine - dropout - softmax
# Conv 계층에서는 폭과 높이가 동일한 필터 사용
# Pooling 계층에서는 2 X 2 풀을 사용
class DeepConvNet:
    # optimizer: 가중치 업데이트 옵티마이저
    # input_shape: 입력 데이터의 형상 - (채널 수, 높이, 폭)
    # conv_param_1 ~ conv_param_6: 각 Conv 계층의 하이퍼파라메터. 필터의 수, 필터의 크기, 보폭, 패딩
    #     filter_num: 필터 수(FN) - Conv 계층 출력의 채널 수
    #     filter_size: 필터의 크기(FH, Fw) - 높이, 폭이 동일한 필터 사용
    #     stride: 보폭
    #     pad: 패딩
    # hidden_size: 은닉층의 수
    # output_size : 출력 크기（MNIST의 경우엔 10）
    def __init__(self, optimizer=Adam(), input_shape=(1, 28, 28),
                 conv_param_1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_2={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_3={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_4={'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1},
                 conv_param_5={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_6={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=50, output_size=10):

        self.optimizer = optimizer
        self.lossValue = None

        ########################################################################
        # Conv 계층의 필터 가중치 및 Affine 계층의 가중치 초기화
        # 각 층의 뉴런 하나당 앞 층의 몇 개 뉴런과 연결되는가（TODO: 자동 계산되게 바꿀 것）
        ########################################################################

        pre_node_nums = np.array(
            [1 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 64 * 3 * 3, 64 * 4 * 4, hidden_size])
        wight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초깃값

        self.params = {}
        pre_channel_num = input_shape[0]
        for idx, conv_param in enumerate(
                [conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            self.params['W' + str(idx + 1)] = wight_init_scales[idx] * np.random.randn(conv_param['filter_num'],
                                                                                       pre_channel_num,
                                                                                       conv_param['filter_size'],
                                                                                       conv_param['filter_size'])
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W7'] = wight_init_scales[6] * np.random.randn(64 * 4 * 4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = wight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)


        ########################################################################
        # 계층 생성
        ########################################################################
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                                       conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                                       conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))

        self.last_layer = SoftmaxWithLoss()



    # 학습
    def train(self, x, t):
        # 기울기 계산
        grads = self.gradient(x, t)

        # 가중치 매개변수 갱신
        self.optimizer.update(self.params, grads)



    # 추론(예측)
    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x



    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        self.lossValue = self.last_layer.forward(y, t)
        return self.lossValue



    # 추론 정확도 계산
    def accuracy(self, x, t, batch_size=100):
        # 정답 레이블이 one-hot-encoding 인 경우 정답 인덱스 추출
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        # 배치처리
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size] # 미니배치 입력
            tt = t[i * batch_size:(i + 1) * batch_size] # 미니배치 정답
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1) # 추론 결과에서 가장 큰 값에 대한 인덱스 추출
            acc += np.sum(y == tt)

        return acc / x.shape[0]


    def gradient(self, x, t):
        ########################################################################
        # Forward - 순전파를 통해 손실을 구한다.(CEE)
        ########################################################################
        self.loss(x, t)

        ########################################################################
        # Backward - 각 계층의 가중치의 미분값 계산
        ########################################################################
        # Softmax - Cross Entropy Error 계층의 미분값 계산
        dout = self.last_layer.backward(1)


        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)


        # 결과 저장
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i + 1)] = self.layers[layer_idx].dW
            grads['b' + str(i + 1)] = self.layers[layer_idx].db

        return grads



    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)



    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i + 1)]
            self.layers[layer_idx].b = self.params['b' + str(i + 1)]





def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


def visualize_filter():

    network = DeepConvNet()

    # 무작위(랜덤) 초기화 후의 가중치
    filter_show(network.params['W1'])

    # 학습된 가중치
    network.load_params("params.pkl")
    filter_show(network.params['W1'])



def trainMNIST():
    from common.mnist import load_mnist
    from common.trainer import Trainer

    # 데이터 읽기
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

    # 시간이 오래 걸릴 경우 데이터를 줄인다.
    # x_train, t_train = x_train[:5000], t_train[:5000]
    # x_test, t_test = x_test[:1000], t_test[:1000]

    max_epochs = 20

    network = DeepConvNet(optimizer=Adam(0.001))
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=max_epochs, batch_size=100,
                      evaluate_sample_num_per_epoch=1000)
    trainer.train()

    # 훈련된 신경망의 매개변수 저장(추후 사용을 위해)
    network.save_params("deep_convnet_params.pkl")
    print("Saved Network Parameters!")

    # 그래프 그리기
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    trainMNIST()

