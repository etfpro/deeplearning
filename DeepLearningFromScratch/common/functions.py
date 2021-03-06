#functions.py

import numpy as np


################################################################################
#
# Activation Functions (활성화 함수들)
#
################################################################################

# ReLU 함수: 은닉층의 활성화 함수
def relu(a):
    return np.maximum(0, a)


# 항등 함수: 출력층의 활성화 함수 - 회귀, 다중 클래스 분류 문제의 추론 시
def identity(a):
    return a

# 시그모이드 함수: 출력층의 활성화 함수 - 2 클래스 분류에 사용
def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


# 소프트맥스 함수" 출력층의 활성화 함수 - 다중 클래스 분류 문제의 학습에 사용, 실제 추론 시에는 속도를 위해 항등 함수 사용
def softmax(a):
    # 데이터가 2개 이상(배치)인 경우
    if a.ndim == 2:
        a = a.T # 최대값이 데이터의 수 만큼의 배열로 출력 되기 때문에 shape을 맞추기 위해서 전치
        a = a - np.max(a, axis=0) # 배열 원소 중 가장 큰 수를 빼서 overflow 방지
        y = np.exp(a) / np.sum(np.exp(a), axis=0)
        return y.T
    else:
        # 데이터가 1개인 경우
        a = a - np.max(a)  # 배열 원소 중 가장 큰 수를 빼서 overflow 방지
        return np.exp(a) / np.sum(np.exp(a))




################################################################################
#
# Loss/Cost Functions (손실/비용 함수들)
# - 손실함수의 리턴값은 스칼라 값
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

    # 신경망의 출력이 1차원(훈련 데이트의 수가 1개)인 경우에도
    # 배치 스타일(행렬) 처리 가능하도록 행렬로 변환
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 정답레이블이 one-hot-encoding 벡터인 경우,
    # 실제 레이블 값 형식으로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)

    # 훈련 데이터의 개수(미니배치 크기)
    batch_size = y.shape[0]

    # np.arange(batch_size): [0, 1, ..., 훈련데이터의 개수 - 1]의 numpy 배열(1차원) 생성
    # ==> y[[0, 1, ..., batch_size - 1], t]: 출력 데이터에서 정답 부분만 추출하여 배열(1차원) 생성
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size




################################################################################
#
# 미분 관련 함수
#
################################################################################

# 수치미분
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h) # 중앙차분



# 기울기 계산 : 모든 변수의 편미분 벡터
# 가중치의 개수 X 2 번 손실함수를 계산하기 때문에 매우 비효율적인 방법
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x) # x의 형상과 같은 배열 생성

    # x의 각 원소에 대한 편미분
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        x_org = x[idx]

        # f(x+h) 계산
        x[idx] = x_org + h
        fxh1 = f(x)  # f(x+h)

        # f(x-h) 계산
        x[idx] = x_org - h
        fxh2 = f(x)

        # 편미분: x[idx] 값의 미분값만 계산, 나머지는 0
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = x_org  # 값 복원

        it.iternext()

    return grad


# 경사하강 기울기 조절
def gradient_descent(f, w, lr=0.01, epoch=100):
    w_history = [] # 기울기 변화 과정을 담은 배열

    for i in range(epoch):
        w_history.append(w.copy())
        w -= lr * numerical_gradient(f, w)

    # 최종 변화된 w와 w의 변화과정을 리턴
    return w, np.array(w_history)



################################################################################
#
# 그래프 함수
#
################################################################################


################################################################################
#
# 테스트
#
################################################################################


if __name__ == '__main__':
    pass