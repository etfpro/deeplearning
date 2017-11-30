#twoLayerNet.py

import numpy as np
import functions as func


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        # 신경망의 매개변수와 편차를 보관하는 dictionary
        self.params = {}

        # hidden layer 가중치 초기화 (표준정규분포 N(0, 1)을 따르는 난수)
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        #self.params['W1'] = np.random.normal(0.0, input_size ** -0.5, (input_size, hidden_size))
        self.params['b1'] = np.zeros(hidden_size)

        # output layer 가중치 초기화 (표준정규분포 N(0, 1)을 따르는 난수)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        #self.params['W2'] = np.random.normal(0.0, hidden_size ** -0.5, (hidden_size, output_size))
        self.params['b2'] = np.zeros(output_size)


    # 예측
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # hidden layer 출력값
        z1 = func.sigmoid(np.dot(x, W1) + b1)

        # output layer 출력값
        y = func.softmax(np.dot(z1, W2) + b2)

        return y

    def forward(self, x):
        return self.predict(x)


    # 손실함수(CEE)
    def loss(self, x, t):
        y = self.predict(x)
        return func.cross_entropy_error(y, t)


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
    def train(self, x, t, learning_rate = 0.01):
        grad = self.numerical_gradient(x, t)
        # grad = networ.gradient(x_batch, t_batch)

        # 매개변수 갱신
        for key in ('W1', 'b1', 'W2', 'b2'):
            self.params[key] -= learning_rate * grad[key]


    # 정확도 측정
    def accuracy(self, x, t):
        return np.sum(np.argmax(self.predict(x), axis=1) == np.argmax(t, axis=1)) / float(x.shape[0])



if __name__ == '__main__':
    net = TwoLayerNet(input_size=784, hidden_size=111, output_size=10)

    x = np.random.rand(100, 784) # 더미 입력 데이터 100개
    t = np.random.rand(100, 10)  # 더미 정답 레이블 100개

    y = net.numerical_gradient(x, t)

    """
    print(grads['W1'].shape)
    print(grads['b1'].shape)
    print(grads['W2'].shape)
    print(grads['b2'].shape)
    """