# coding: utf-8
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.functions import numerical_gradient



# 단순한 합성곱 신경망
# conv - relu - pool - affine - relu - affine - softmax
# Conv 계층에서는 폭과 높이가 동일한 필터 사용
# Pooling 계층에서는 2 X 2 풀을 사용
class SimpleConvNet:
    # input_channel: 입력 데이터의 채널 수
    # input_size: 입력 데이터의 크기(H, W) - 높이, 폭이 동일한 데이터라고 가정
    # conv_param: Conv 계층의 하이퍼파라메터. 필터의 수, 필터의 크기, 보폭, 패딩
    #     filter_num: 필터 수(FN) - Conv 계층 출력의 채널 수
    #     filter_size: 필터의 크기(FH, Fw) - 높이, 폭이 동일한 필터 사용
    #     stride: 보폭
    #     pad: 패딩
    # hidden_size: 은닉층의 수
    # output_size : 출력 크기（MNIST의 경우엔 10）
    # weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
    #     'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
    #     'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    def __init__(self, input_channel = 1, input_size=28,
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std='0.01'):
        # Conv 계층의 하이퍼파라메터
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']

        # Pool의 크기(동일한 높이, 폭 사용)
        pool_size = 2

        # Conv계층의 출력 크기(OH, OW) 계산
        conv_output_size = int((input_size + 2 * filter_pad - filter_size) / filter_stride + 1)

        # Pooling계층의 출력 크기 계산
        # 2 X 2, 보폭 2인 풀 사용
        pool_output_size = int(filter_num * (conv_output_size / pool_size) * (conv_output_size / pool_size))


        ########################################################################
        # Conv 계층의 필터 가중치 및 Affine 계층의 가중치 초기화
        ########################################################################
        self.params = {}

        # Conv 계층의 필터 가중치 초기화(FN, C, FH, FW)
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_channel, filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)

        # Affine 계층의 가중치
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)


        ########################################################################
        # 계층 생성
        ########################################################################
        self.layers = OrderedDict()

        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           filter_stride, filter_pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=pool_size, pool_w=pool_size, stride=pool_size)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()


    # 추론(예측)
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x



    def loss(self, x, t):
        """손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """기울기를 구한다（수치미분）.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

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

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]
