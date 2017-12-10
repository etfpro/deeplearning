#twoLayerNet.py

import numpy as np
import functions as func
from layers import *
from collections import OrderedDict
from optimizer import Momentum


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, optimizer=Momentum()):


        ########################################################################
        # 신경망의 매개변수(가중치, 편차)초기화
        ########################################################################
        self.params = {}

        # hidden layer 가중치 초기화 - He 초기값
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        #self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        #self.params['W1'] = np.random.randn(input_size, hidden_size) * 0.01
        self.params['b1'] = np.zeros(hidden_size)

        # output layer 가중치 초기화 - He 초기값
        self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        #self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        #self.params['W2'] = np.random.randn(hidden_size, output_size) * 0.01
        self.params['b2'] = np.zeros(output_size)


        ########################################################################
        # 각 계층 생성
        ########################################################################
        self.layers = OrderedDict()

        # hiden layer
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()

        # output layer
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss() # Softmax와 오차함수 계층은 실제 추론에서는 사용하지 않기 때문에 별도의 변수에 저장


        ########################################################################
        # 가중치 갱신 optimizer
        ########################################################################
        self.optimizer = optimizer



    # 추론(예측)
    # hidden layer ~ output layer의 Affine 계층까지
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x



    # 손실함수(CEE): 순전파를 통한 손실을 구한다
    # x: 입력 데이터
    # t: 정답 레이블
    def loss(self, x, t):
        # hidden layer ~ output layer의 Affine 계층
        a = self.predict(x)

        # Softmax - Cross Entropy Error 계층
        return self.lastLayer.forward(a, t)



    # 손실함수의 기울기 계산(미분)
    # x: 입력 데이터
    # t: 정답 레이블
    def gradient(self, x, t):
        # 순전파를 통해 손실을 구한다.(CEE)
        # 손실값은 SoftmaxWithLoss 계층 객체에 저장
        self.loss(x, t)

        # 미분값 계산

        # Softmax - Cross Entropy Error 계층의 미분값 계산
        dout = self.lastLayer.backward(1.0)

        # output layer의 Affine 계층 ~ hidden layer 까지 미분값 계산
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 미분 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


    # 손실함수의 기울기 계산: 손실함수를 가중치에 대해서 수치 미분
    # 계산 시간이 오래 걸린다
    # x: 입력 데이터
    # t: 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        # 모든 레이어의 가중치를 마지막 출력층의 손실함수를 미분한 값으로 갱신
        grads['W1'] = func.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = func.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = func.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = func.numerical_gradient(loss_W, self.params['b2'])

        return grads


    # 학습
    def train(self, x, t):
        grads = self.gradient(x, t)

        # 매개변수 갱신
        self.optimizer.update(self.params, grads)


    # 정확도 측정
    def accuracy(self, x, t):
        # 출력값 중 가장 큰 값의 인덱스 추출
        y = np.argmax(self.predict(x), axis=1)

        # 정답 레이블이 one-hot-encoding 인 경우 정답 인덱스(값이 1) 추출
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])


if __name__ == '__main__':
    net = TwoLayerNet(input_size=784, hidden_size=111, output_size=10)

    x = np.random.rand(100, 784) # 더미 입력 데이터 100개
    t = np.random.rand(100, 10)  # 더미 정답 레이블 100개


    """
    print(grads['W1'].shape)
    print(grads['b1'].shape)
    print(grads['W2'].shape)
    print(grads['b2'].shape)
    """