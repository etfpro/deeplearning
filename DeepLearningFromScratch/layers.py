# Layers.py
# 역전파 시 해당 함수를 국소적 미분한 값을 곱한다.

import numpy as np
import functions as func


# Affine 변환 계층 (X·W + b)
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        # 순전파 입력을 역전파(미분) 시 사용을 위해 저장
        self.x = None

        # 가중치에 대한 미분값 (이전 계층으로 전달하지 않고 내부에서만 유지)
        self.dW = None

        # 편차에 대한 미분값 (이전 계층으로 전달하지 않고 내부에서만 유지)
        self.db = None

        # 입력값의 형상(텐서 대응)
        self.original_x_shape = None


    #  입력 - x는 이전 레이어의 출력값
    def forward(self, x):
        self.original_x_shape = x.shape

        # 입력을 행렬(2차원 배열)로 변환
        # 행: 데이터의 수, 열: 1개의 입력값을 1행으로 풀어놓은 상태
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b
        return out


    # Affine 함수 미분
    # 입력 - dout은 다음 계층(활성화 함수 계층)의 미분 값
    # 출력 - 이전 계충(활성화함수 계층)으로 전파할 값
    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = np.dot(dout, self.W.T)
        # 행렬(2차원 배열) 형태의 입력 데이터를 원래의 형상으로 변환(텐서 대응)
        dx = dx.reshape(*self.original_x_shape)
        return dx



# ReLU 활성화 함수 계층
# 은닉계층의 활성화 함수로 주로 사용
class Relu:
    def __init__(self):
        # 입력값 중  0아하 인 것들을 나타내는 배열
        # 추후 역전파를 위한 미분에 사용
        self.mask = None


    # 입력 - x는 Affine 계층의 출력값
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy() # 입력 값을 변화시키지 않기 위해 복사
        out[self.mask] = 0 # 0 이하인 것들에 대해 0으로 변경
        return out


    # ReLU 함수 미분
    # 0보다 작거나 같은 경우 0, 0보다 큰 경우 1
    # 입력 - dout은 다음 계층(Affine 계층)의 미분 값
    # 출력 - 이전 계충(Affine 계증)으로 전파할 값
    def backward(self, dout):
        dout[self.mask] = 0 # 0 이하인 것들에 대해 0으로 변경
        dx = dout
        return dx



# Sigmoid 활성화 함수 계층
# 2진 분류 시 출력층의 활성화 함수로 주로 사용
class Sigmoid:
    def __init__(self):
        self.out = None


    # 입력 - x는 Affine 계층의 출력값
    def forwward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-x))
        out = self.out
        return out


    # Sigmoid 함수 미분
    # y*(1-y)
    # 입력 - dout은 다음 계층의 미분 값
    # 출력 - 이전 계충(Affine계층)으로 전파할 값
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx;



# Softmax 활성화 함수 + Cross Entropy Error 함수 계층
# 분류 신경망의 마지막 계층으로 학습시에만 사용된다.
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실
        self.y = None # Softamx 출력
        self.t = None # 정답 레이블

    # 입력값 x는 Affine 계층의 출력값
    # t: 정답 레이블(마지막 계층이기 때문에 정답 레이블이 필요)
    def forward(self, x, t):
        self.t = t
        self.y = func.softmax(x)
        self.loss = func.cross_entropy_error(self.y, self.t)
        return self.loss


    # 입력 - dout은 마지막 계층이기 때문에 1
    # 출력 - 이전 계충(Affine 계층)으로 전파할 값
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]

        # 정답 레이블이 one-hot encoding 형태인 경우
        if self.t.size == self.y.size:
            dx = self.y - self.t
        else:
            # 정답 레이블이 레이블 형태인 경우
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1 # 정답 레이블에 해당하는 항목에서만 1(정답 확률)을 뺀다

        # 배치의 수로 나눠서 데이터 1개당 오차를 전파 ????????????????
        dx /= batch_size
        return dx



class Layer:
    def __init__(self, W, b, activation=Relu()):
        self.affine = Affine(W, b)
        self.activation = activation


    def forward(self, x):
        x = self.affine.forward(x)
        if self.activation != None:
            x = self.activation.forward(x)
        return x

    def backward(self, dout):
        if self.activation != None:
            dout = self.activation.backward(dout)
        dout = self.affine.backward(dout)
        return dout


    def dW(self):
        return self.affine.dW


    def db(self):
        return self.affine.db



if __name__ == '__main__':
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])



