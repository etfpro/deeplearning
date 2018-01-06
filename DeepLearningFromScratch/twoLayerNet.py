#twoLayerNet.py

import numpy as np
import functions as func
from layers import *
from collections import OrderedDict
from optimizer import SGD
from mnist import load_mnist


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, optimizer=SGD()):

        ########################################################################
        # 신경망의 매개변수(가중치, 편차) 초기화
        ########################################################################
        self.params = {}

        # hidden layer 가중치 초기화
        #self.params['W1'] = np.random.randn(input_size, hidden_size) * 0.01
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        #self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        #self.params['W1'] = np.random.randn(input_size, hidden_size) * 0.01
        self.params['b1'] = np.zeros(hidden_size)

        # output layer 가중치 초기화
        #self.params['W2'] = np.random.randn(hidden_size, output_size) * 0.01
        self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        #self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        #self.params['W2'] = np.random.randn(hidden_size, output_size) * 0.01
        self.params['b2'] = np.zeros(output_size)


        ########################################################################
        # 각 계층 생성
        ########################################################################
        self.layers = OrderedDict()

        # hiden layer
        self.layers['Hidden'] = Layer(self.params['W1'], self.params['b1'])

        # output layer
        self.layers['Output'] = Layer(self.params['W2'], self.params['b2'], None)
        self.lossLayer = SoftmaxWithLoss() # Softmax와 오차함수 계층은 실제 추론에서는 사용하지 않기 때문에 별도의 변수에 저장


        ########################################################################
        # 기타
        ########################################################################
        #가중치 갱신 optimizer
        self.optimizer = optimizer

        # 가중치 갱신 optimizer
        self.lossValue = 0


    # 추론(예측)
    # hidden layer ~ output layer의 Affine 계층까지
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x



    # 손실함수(CEE)
    # x: 입력 데이터
    # t: 정답 레이블
    # 출력은
    def loss(self, x, t):
        # hidden layer ~ output layer의 Affine 계층
        a = self.predict(x)

        # Softmax - Cross Entropy Error 계층
        self.lossValue = self.lossLayer.forward(a, t)
        return self.lossValue



    # 정확도 측정
    def accuracy(self, x, t):
        # 출력값 중 가장 큰 값의 인덱스 추출
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        # 정답 레이블이 one-hot-encoding 인 경우 정답 인덱스 추출
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0]) # 배치 크기



    # 손실함수의 기울기 계산(미분)
    # 오차역전파를 통해 각 가중치 매개변수에 대한 미분값 계산
    # x: 입력 데이터
    # t: 정답 레이블
    def gradient(self, x, t):
        # 순전파를 통해 손실을 구한다.(CEE)
        # 손실값은 SoftmaxWithLoss 계층 객체에 저장
        self.loss(x, t)

        # 미분값 계산

        # Softmax - Cross Entropy Error 계층의 미분값 계산
        dout = self.lossLayer.backward(1.0)

        # output layer의 Affine 계층 ~ hidden layer 까지 미분값 계산
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 미분 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Hidden'].dW(), self.layers['Hidden'].db()
        grads['W2'], grads['b2'] = self.layers['Output'].dW(), self.layers['Output'].db()

        return grads



    # 학습
    def train(self, x, t):
        # 기울기 계산
        grads = self.gradient(x, t)

        # 가중치 매개변수 갱신
        self.optimizer.update(self.params, grads)



    # 손실함수의 기울기 계산
    # 손실함수를 가중치에 대해서 수치 미분
    # 계산 시간이 오래 걸리기 때문에 실제 기울기 계산이 아닌 오차역전파에 의해 계산한 기울기 검증에 사용
    # x: 입력 데이터
    # t: 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}

        # 모든 레이어의 기울기를 손실함수를 미분한 값으로 계산
        grads['W1'] = func.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = func.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = func.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = func.numerical_gradient(loss_W, self.params['b2'])

        return grads




################################################################################
#
# 테스트
#
################################################################################

def gradientCheck():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    x_batch = x_train[:3]
    t_batch = t_train[:3]

    # 수치미분에 의한 기울기 계산
    grad_numerical = network.numerical_gradient(x_batch, t_batch)

    # 오차역전파에 의한 기울기 계산
    grad_backprop = network.gradient(x_batch, t_batch)

    # W1, b1, W2, b2에 대해서 수치미분과 오차역전파에 의한 기울기 계산값의 차이의 평균 계산
    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_numerical[key] - grad_backprop[key]))
        print(key + ":" + str(diff))




if __name__ == '__main__':
    gradientCheck()