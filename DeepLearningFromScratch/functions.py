#start.py

import numpy as np
import matplotlib.pylab as plt
import os


################################################################################
#
# Activation Functions (활성화 함수들)
#
################################################################################

def identity(x):
    return x

def step_func(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # 훈련 데이터가 2개 이상인 경우
    if x.ndim == 2:
        x = x.T # 최대값이 훈련 데이터의 수 만큼의 배열로 출력 되기 때문에 shape을 맞추기 위해서 전치
        x = x - np.max(x, axis=0) # 배열 원소 중 가장 큰 수를 빼서 overflow 방지
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    else:
        x = x - np.max(x)  # 배열 원소 중 가장 큰 수를 빼서 overflow 방지
        return np.exp(x) / np.sum(np.exp(x))



################################################################################
#
# Loss/Cost Functions (손실/비용 함수들)
#
################################################################################

# 평균제곱오차(MSE) 함수
#  y: 신경망 출력
#  t: 정답레이블
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


# 교차엔트로피오차(CEE) 함수
#  y: 신경망 출력
#  t: 정답레이블
def cross_entropy_error(y, t):

    # 신경망의 출력이 1차원(출력이 1개)인 경우에도 처리 가능하도록 행열로 변환
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 정답레이블이 one-hot-encoding 벡터인 경우,
    # 정답레이블에서 최대 인덱스를 구해서 실제 레이블 값 형식의 배열(1차원)으로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0] # 훈련데이터의 개수(미니 배치 크기

    # np.arange(batch_size): [0, 1, ..., 훈련데이터의 개수 - 1]의 numpy 배열(1차원) 생성
    # ==> y[[0, 1, ..., batch_size - 1], t]: 출력 데이터에서 정답 부분만 추출하여 배열(1차원) 생성
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size




################################################################################
#
# 미분 관련 함수
#
################################################################################

# 수치미분
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h) # 중앙차분



# 손실함수를 가중치로 미분하여 기울기 계산
# 가중치의 개수 X 2 번 손실함수를 계산하기 때문에 매우 비효율적인 방법
def numerical_gradient(lossFunc, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x) # x의 형상과 같은 배열 생성

    # x의 각 원소에 대한 편미분
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        x_org = x[idx]

        # f(x+h) 계산
        x[idx] = x_org + h
        fxh1 = lossFunc(x)  # f(x+h)

        # f(x-h) 계산
        x[idx] = x_org - h
        fxh2 = lossFunc(x)

        # 미분
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = x_org  # 값 복원
        it.iternext()

    return grad


# 경사하강 기울기 조절
def gradient_descent(f, x, lr=0.01, epoch=100):
    w_history = [] # 기울기 변화 과정을 담은 배열

    for i in range(epoch):
        w_history.append(x.copy())
        x -= lr * numerical_gradient(f, x)

    return x, np.array(w_history)




################################################################################
#
# 테스트
#
################################################################################


if __name__ == '__main__':
    pass
