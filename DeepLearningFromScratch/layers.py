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



# 배치 정규화 계층
# 활성화값을 적당히 분산시키기 위해 사용
# 학습속도 개선, 기울기 감소 개선, 초기값 의존성 감소, 오버피팅 억제(드롭아웃 필요성 감소)
# Affine 계층과 Activation(ReLU) 계층 사이에 존재
# z = (x - 평균) / 표준편차
# y = gamma * z + beta
# Training 할 때는 mini-batch의 평균과 분산으로 normalize 하고,
# Test(추론) 할 때는 계산해 놓은 이동평균으로 normalize 한다.
# Normalize 한 이후에는 scale factor(gamma)와 shift factor(beta)를 이용하여 새로운 값을 만들고, 이 값을 출력
# 이 Scale factor와 Shift factor는 다른 레이어에서 weight를 학습하듯이 역전파에서 학습
# http://arxiv.org/abs/1502.03167
class BatchNormalization:
    # gmma - Scale factor(초기값 1)
    # beta - Shift factor(초기값 0)
    # momentum
    def __init__(self, gamma, beta, momentum=0.9): # running_mean=None, running_var=None
        self.gamma = gamma # Scale factor
        self.beta = beta # Shift factor

        # 4차원 합성곱을 2차원으로 reshape 후, 다시 4차원으로 복구하기 위해 저장
        self.input_shape = None

        # 시험(추론)할 때 사용할 평균과 분산 - Training 시에 미니배치에서 구한 평균과 분산의 이동평균
        self.running_mean = None
        self.running_var = None
        self.momentum = momentum # 이동평균 계산 시 사용할 momentum

        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None # 편차
        self.std = None # 표준편차
        self.dgamma = None # Sacel factor 미분값
        self.dbeta = None # Shift factor 미분값


    # 입력 - x는 Affine 계층의 출력값
    def forward(self, x, train_flg=True):
        self.input_shape = x.shape

        # 4차원 합성곱 데이터는 2차원으로 변경
        if x.ndim != 2:
            N, _, _, _ = x.shape
            x = x.reshape(N, -1)

        # 순전파 실행
        out = self.__forward(x, train_flg)

        # 원래의 입력 데이터 차원으로 다시 복원
        return out.reshape(*self.input_shape)


    # 입력 - x는 Affine 계층의 출력값
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            # Training - 미니배치의 평균과 분산으로 정규화

            # 미니배치의 평균 계산
            mu = x.mean(axis=0) # 미니배치 데이터 평균 - 동일한 열의 평균

            # 미니배치의 분산 계산
            xc = x - mu
            var = np.mean(xc ** 2, axis=0) # 동일한 열의 분산

            # 미니배치의 표준값 계산
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            # 역전파(backward) 시 사용할 변수 저장
            self.batch_size = x.shape[0] # 미니배치 수
            self.xc = xc # 편차
            self.xn = xn # 표준값
            self.std = std # 표준편차

            # 테스트(추론) 시 사용할 평균과 분산에 대한 지수 이동평균(Moving Average) 계산
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # 테스트(추론) - 미니배치를 사용할 수 없기 때문에 training에 계산한 이동평균과 분산으로 정규화
            xc = x - self.running_mean # 편차
            xn = xc / ((np.sqrt(self.running_var + 10e-7))) # 표준값

        # Scale & Shift
        out = self.gamma * xn + self.beta
        return out


    def backward(self, dout):
        # 4차원 합성곱 데이터는 2차원으로 변경
        if dout.ndim != 2:
            N, _, _, _ = dout.shape
            dout = dout.reshape(N, -1)

        # 역전파 실행
        dx = self.__backward(dout)

        # 원래의 입력 데이터 차원으로 다시 복원
        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        # Scale factor 미분
        self.dgamma = np.sum(self.xn * dout, axis=0)

        # Shift factor 미분
        self.dbeta = dout.sum(axis=0)

        # 이전 계층으로 전달할 미분
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        return dx



# ReLU 활성화 함수 계층
# 은닉계층의 활성화 함수로 주로 사용
class Relu:
    def __init__(self):
        # 입력값 중  0아하 인 것들을 나타내는 배열
        # 추후 역전파를 위한 미분에 사용
        self.mask = None


    # 입력 - x는 Affine 계층 또는 BatchNormalization 계층의 출력값
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
    pass



