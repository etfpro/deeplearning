import numpy as np

# 확률적 경사하강법
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr


    # 가중치 업데이트(1회)
    # params: 업데이트할 가중치 행렬들
    # grads: 가중치에 대한 미분값(기울기) 행렬
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# Momentum
# Gradient Descent를 통해 이동하는 과정에 일종의 관성/탄력을 주는 방식
# 현재의 기울기를 통해 이동하는 방향과는 별개로, 과거에 이동했던 방식을 기억하면서 그 방향으로 일정 정도를 추가적으로 이동하는 방식
# 자주 이동하는 방향에 관성이 걸리게 되고, 진동을 하더라도 중앙으로 가는 방향에 힘을 얻기 때문에 SGD에 비해 상대적으로 빠르게 이동할 수 있다.
# v = momentum * v + learing_rate * 기울기, W = W - v (v 초기값은 0)
# momentum이 1 미만이기 때문에 학습이 진행될수록 과거의 갱신량은 점점 작아지게 된다
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None


    # 가중치 업데이트(1회)
    # params: 업데이트할 가중치 행렬들
    # grads: 가중치에 대한 미분값(기울기) 행렬
    def update(self, params, grads):
        # 최초에는 업데이트할 가중치와 동일한 형상의 행렬로 초기화(초기값은 0)
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 가중치 업데이트
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] + self.lr * grads[key]
            params[key] -= self.v[key]



# AdaGrad - Adaptive Gradients, 학습률 감소 기법
# 가중치를 update할 때 각각의 가중치마다 학습률을 다르게 설정해서 이동
# 기본적인 아이디어는 '지금까지 적게 갱신된 가중치들은 학습률을 크게 하고, 지금까지 많이 갱신된 가중치들은 학습률을 작게 하자’라는 것
# 자주 등장하거나 변화를 많이 한 가중치들의 경우 최적에 가까이 있을 확률이 높기 때문에, 가중치를 작게 하여 작은 크기로 이동하면서 세밀한 값을 조정하고,
# 자주 등장하지 않거나 적게 변화한 가중치들은 최적 값에 도달하기 위해서는 많이 이동해야할 확률이 높기 때문에, 먼저 빠르게 손실값을 줄이는 방향으로 이동하려는 방식
# h = h + 기울기^2, W = W - learning_rate/sqrt(h) * 기울기 (h 초기값은 0)
# 과거의 기울기를 제곱하여 계속 더해가기 대문에, 학습을 진행할수록 갱신 강도가 약해지는 문제가 있기 때문에, 무한히 계속 학습한다면 어느 순간 갱신량이 0이되어
# 전혀 갱신되지 않는 문제가 발생
# 이 문제를 개선하기 위해서 먼 과거의 기울기는 서서히 잊고 새로운 기울기 정보를 크게 반영(지수이동평균)하는 RMSProp 기법이 있다.
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None


    # 가중치 업데이트(1회)
    # params: 업데이트할 가중치 행렬들
    # grads: 가중치에 대한 미분값(기울기) 행렬
    def update(self, params, grads):
        # 최초에는 업데이트할 가중치와 동일한 형상의 행렬로 초기화(초기값은 0)
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        # 가중치 업데이트
        for key in params.keys():
            self.h[key] += grads[key] ** 2
            params[key] -= (self.lr / (np.sqrt(self.h[key]) + 1e-7)) * grads[key]


# RMSprop - AdaGrad 개선 버전, 학습률 감소
# AdaGrad는 과거의 기울기를 제곱해서 계속 더해가기 때문에, 학습을 진행할 수록 갱신 강도가 약해짐
# 실제로 무한히 계속 학습한다면 어느 순간 갱신량이 0이되어 전혀 갱신되지 않은 문제가 있음
# 따라서 과거의 모든 기울기를 균일하게 더해가는 것이 아니라, 먼 과거의 기울기는 서서히 잊고 새로운 기울기 정보를 크게 반영 - 지수이동평균
# 즉, 과거 기울기의 반영 규모를 기하급수적으로 감소시킴
# 강화학습에서 많이 사용
class RMSProp:
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            # 이전 기울기를 감소시킨다.
            self.h[key] = self.decay_rate * self.h[key] + (1 - self.decay_rate) * grads[key] ** 2
            params[key] -= (self.lr / (np.sqrt(self.h[key]) + 1e-7)) * grads[key]



# Adaptive Moment Estimation: Mementum(관성) + RMSProp(학습률 감소) 기법 혼합
# 이 방식에서는 Momentum 방식과 유사하게 지금까지 계산해온 기울기의 지수평균을 저장하며, RMSProp과 유사하게 기울기의 제곱값의 지수평균을 저장한다.
class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1 # 1차 momentum용 계수
        self.beta2 = beta2 # 2차 momentum용 계수
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from collections import OrderedDict

    # 손실함수
    def f(x, y):
        return x**2 / 20.0 + y**2

    # 위의 손실함수 f(x, y) 함수 미분
    def df(x, y):
        return x / 10.0, 2.0 * y

    # optimizer들
    optimizers = OrderedDict()
    optimizers["SGD"] = SGD(lr=0.95)
    optimizers["Momentum"] = Momentum(lr=0.1)
    optimizers["AdaGrad"] = AdaGrad(lr=1.5)
    optimizers["Adam"] = Adam(lr=0.3)


    # 가중치 매개변수 x, y
    params = {}

    # 가중치 x, y에 대한 기울기
    grads = {}
    grads['x'], grads['y'] = 0, 0

    idx = 1

    # 각 optimizer 별로 기울기 갱신
    for key in optimizers:
        optimizer = optimizers[key]

        # 가중치 매개변수 x, y의 변화를 저장
        x_history, y_history = [], []

        # 가중치 매개변수 x, y 초기값
        params['x'], params['y'] = -7.0, 2.0

        # 매개변수 갱신(경사하강) 수행
        for i in range(30):
            # 이전 가중치 매개변수 x, y 값 저장
            x_history.append(params['x'])
            y_history.append(params['y'])

            # 가중치 매개변수 x, y에 대한 미분-기울기 계산
            grads['x'], grads['y'] = df(params['x'], params['y'])

            # 가중치 매개변수 x, y를 기울기에 의해 갱신
            optimizer.update(params, grads)

        # 가중치 매개변수 (x, y) 쌍에 대한 손실 함수값 Z 계산
        x = np.arange(-12, 12, 0.01)
        y = np.arange(-4, 4, 0.01)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        # 손실함수 값 중 7보다 큰 값은 0으로 변경하여 0~7 사이의 값만 표시
        mask = Z > 7
        Z[mask] = 0

        # 그래프 그리기
        plt.subplot(2, 2, idx)
        idx += 1
        plt.plot(x_history, y_history, 'o-', color="red")
        plt.plot(0, 0, '+')
        plt.contour(X, Y, Z) # 등고선 그리기
        plt.xlim(-12, 12)
        plt.ylim(-5, 5)
        # colorbar()
        # spring()
        plt.title(key)
        plt.xlabel("x")
        plt.ylabel("y")

    plt.show()
