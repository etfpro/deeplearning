# multiLayerNet.py
from common.layers import *
from common.optimizer import *
from collections import OrderedDict



# Fully Connected 심층 신경망
class MultiLayerNet:

    # input_size : 입력 크기（MNIST의 경우엔 784）
    # hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    # output_size : 출력 크기（MNIST의 경우엔 10）
    # activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    # weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
    #                   'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
    #                   'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    # use_batchnorm : 배치 정규화 사용 여부
    # weight_decay_lambda : 가중치 감소(L2 법칙)의 세기, 0이면 가중치 감소를 수행하지 않음
    # dropout_ration : dropout 비율, 0이면 dropout을 수행하지 않음
    def __init__(self, input_size, hidden_size_list, output_size, optimizer=Adam(),
                 activation='relu', weight_init_std='relu', use_batchnorm=False,
                 weight_decay_lambda=0, dropout_ratio=0):

        ########################################################################
        # 각종 속성
        ########################################################################
        #가중치 갱신 optimizer
        self.optimizer = optimizer

        # 손실값
        self.lossValue = 0

        # 배치정규화 사용 여부
        self.use_batchnorm = use_batchnorm

        # 가중치 감소 비율
        self.weight_decay_lambda = weight_decay_lambda

        # hidden layer의 수
        self.hidden_layer_num = len(hidden_size_list)


        ########################################################################
        # 신경망의 매개변수(가중치, 편차) 초기화
        ########################################################################
        self.params = {}
        self.__init_weight(input_size, hidden_size_list, output_size, weight_init_std)


        ########################################################################
        # 각 계층 생성 및 초기화
        ########################################################################
        self.layers = OrderedDict()
        self.__init_layers(activation, hidden_size_list, dropout_ratio)


    # 가중치 초기화
    # weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
    #                  'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
    #                  'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    def __init_weight(self, input_size, hidden_size_list, output_size, weight_init_std):
        # 각 layer별 크기에 대한 리스트
        all_size_list = [input_size] + hidden_size_list + [output_size]

        # 첫번째 hidden layer ~ output layer 까지의 가중치들에 대한 초기화
        for i in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'): # ReLU를 사용할 때의 권장 초기값 - HE
                scale = np.sqrt(2.0 / all_size_list[i - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'): # sigmoid를 사용할 때의 권장 초기값 - XAVIER
                scale = np.sqrt(1.0 / all_size_list[i - 1])

            self.params['W' + str(i)] = scale * np.random.randn(all_size_list[i - 1], all_size_list[i])
            self.params['b' + str(i)] = np.zeros(all_size_list[i])


    # 각 계층 생성 및 초기화
    def __init_layers(self, activation, hidden_size_list, dropout_ratio):
        ########################################################################
        # hidden layers
        ########################################################################
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu} # 활성화 계층
        for i in range(1, self.hidden_layer_num + 1):
            # Affine layer
            self.layers['Affine' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])

            # Batch Normalization Layer
            if self.use_batchnorm:
                self.params['gamma' + str(i)] = np.ones(hidden_size_list[i - 1]) # Scale factor (초기값 1)
                self.params['beta' + str(i)] = np.zeros(hidden_size_list[i - 1]) # Shift factor (초기값 0)
                self.layers['BatchNorm' + str(i)] = BatchNormalization(self.params['gamma' + str(i)], self.params['beta' + str(i)])

            # Activation Layer
            self.layers['Activation' + str(i)] = activation_layer[activation]()

            # Dropout Layer
            if dropout_ratio > 0:
                self.layers['Dropout' + str(i)] = Dropout(dropout_ratio)


        ########################################################################
        # output layer
        ########################################################################
        i = self.hidden_layer_num + 1
        self.layers['Affine' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])
        self.last_layer = SoftmaxWithLoss() # Softmax와 오차함수 계층은 실제 추론에서는 사용하지 않기 때문에 별도의 변수에 저장



    # 추론(예측)
    # hidden layer ~ output layer의 Affine 계층까지
    def predict(self, x, train_flg):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x



    # 손실함수(CEE)
    # x: 입력 데이터
    # t: 정답 레이블
    def loss(self, x, t, train_flg):
        # hidden layer ~ output layer의 Affine 계층
        a = self.predict(x, train_flg)

        # 가중치감소 처리(가중치감소값 = 모든 가중치의 제곱 합 * 람다 / 2)
        weight_decay = 0
        if (self.weight_decay_lambda > 0):
            weight_square_sum = 0
            for i in range(1, self.hidden_layer_num + 2):
                W = self.params['W' + str(i)]
                weight_square_sum += np.sum(W ** 2)
            weight_decay = 0.5 * self.weight_decay_lambda *  weight_square_sum

        # Softmax - Cross Entropy Error 계층
        self.lossValue = self.last_layer.forward(a, t) + weight_decay



    # 정확도 측정
    def accuracy(self, x, t):
        # 출력값 중 가장 큰 값의 인덱스 추출
        y = self.predict(x, train_flg=False)
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
        ########################################################################
        # Forward - 순전파를 통해 손실을 구한다.(CEE)
        # 손실값은 SoftmaxWithLoss 계층 객체에 저장
        ########################################################################
        self.loss(x, t, train_flg=True)


        ########################################################################
        # Backward - 각 계층의 가중치의 미분값 계산
        ########################################################################
        # Softmax - Cross Entropy Error 계층의 미분값 계산
        dout = self.last_layer.backward(1.0)

        # output layer의 Affine 계층 ~ 첫번째 hidden layer 까지 미분값 계산
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 미분결과(기울기) 저장
        grads = {}
        for i in range(1, self.hidden_layer_num + 2): # 첫번째 hidden layer ~ output layer
            grads['W' + str(i)] = self.layers['Affine' + str(i)].dW + \
                                  self.weight_decay_lambda * self.layers['Affine' + str(i)].W # 가중치감소 미분값(람다 * W)을 더한다
            grads['b' + str(i)] = self.layers['Affine' + str(i)].db

            if self.use_batchnorm and i != self.hidden_layer_num + 1:
                grads['gamma' + str(i)] = self.layers['BatchNorm' + str(i)].dgamma
                grads['beta' + str(i)] = self.layers['BatchNorm' + str(i)].dbeta

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
        loss_W = lambda W: self.loss(x, t, train_flg=True)

        grads = {}
        for i in range(1, self.hidden_layer_num + 2):
            grads['W' + str(i)] = func.numerical_gradient(loss_W, self.params['W' + str(i)])
            grads['b' + str(i)] = func.numerical_gradient(loss_W, self.params['b' + str(i)])

            # 배치 정규화 사용 시 (마지막 출력층에는 배치 정규화 계층이 존재하지 않음)
            if self.use_batchnorm and i != self.hidden_layer_num + 1:
                grads['gamma' + str(i)] = func.numerical_gradient(loss_W, self.params['gamma' + str(i)])
                grads['beta' + str(i)] = func.numerical_gradient(loss_W, self.params['beta' + str(i)])

        return grads




if __name__ == '__main__':
    pass