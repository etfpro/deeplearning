# neuralNetwork Class

import numpy as np

class neuralNetwork:

    # 신경망 초기화
    def __init__(self, inputnodes, outputnodes, learningrate, debugMode=False):
        # 입력 노드 수
        self.inodes = inputnodes

        # 은닉 노드 수
        self.hnodes = max(inputnodes, inputnodes // 5)

        # 출력 노드 수
        self.fnodes = outputnodes

        # 학습률
        self.lr = learningrate

        # 은닉층 가중치
        self.hidden_weights = np.random.normal(0.0, self.inodes ** -0.5, (self.inodes, self.hnodes))

        # 출력층 가중치
        self.final_weights = np.random.normal(0.0, self.hnodes ** -0.5, (self.hnodes, self.fnodes))

        # 활성화 함수
        self.activation_function = lambda x: 1.0 / (1.0 + np.exp(-x))

        # 디버그모드
        self.debugMode = debugMode



    # 가중치 업데이트
    def updateHiddenWeights(self, weights, errors, inputs, outputs):
        deltaW = self.lr * -np.dot(inputs.T, errors * outputs * (1 - outputs))
        weights -= deltaW

    def updateOutputWeights(self, weights, errors, inputs, outputs):
        deltaW = self.lr * -np.dot(inputs.T, errors * outputs * (1 - outputs))
        weights -= deltaW


    # 신경망 학습
    def train(self, training_data, training_label):
        # 학습데이터와 레이블(정답)을 행렬로 변환
        inputs = np.array(training_data, ndmin=2)
        labels = np.array(training_label, ndmin=2)

        if self.debugMode:
            print(">> training_data <<\n", inputs)

        ########################################################################
        # forward propagation
        ########################################################################

        # 은닉층
        hidden_outputs = self.activation_function(np.dot(inputs, self.hidden_weights))

        # 출력층
        final_outputs = self.activation_function(np.dot(hidden_outputs, self.final_weights))
        if self.debugMode:
            print(">> Final Outputs(training...) <<\n", final_outputs)


        ########################################################################
        # back propagation(오차 역전파)
        ########################################################################

        # 출력층 오차 (레이블 - 계산값)
        final_errors = labels - final_outputs
        #if self.debugMode:
        #   print(">> Final Errors(training...) <<\n", final_errors)

        # 은닉층 오차 (출력층 오차 X 출력층 가중치의 전치행렬)
        hidden_errors =  np.dot(final_errors, self.final_weights.T)
        #if self.debugMode:
        #   print(">> Hidden Errors(training...) <<\n", hidden_errors)


        ########################################################################
        # 가중치 업데이트
        ########################################################################

        # 출력층 가중치 업데이트
        #if self.debugMode:
        #    print(">> Final Weiths Before Update(training...) <<\n", self.final_weights)
        self.updateOutputWeights(self.final_weights, final_errors, hidden_outputs, final_outputs)
        #if self.debugMode:
        #    print(">> Final Weiths After Update(training...) <<\n", self.final_weights)

        # 은닉층 가중치 업데이트
        #if self.debugMode:
        #    print(">> Hidden Weiths Before Update(training...) <<\n", self.hidden_weights)
        self.updateHiddenWeights(self.hidden_weights, hidden_errors, inputs, hidden_outputs)
        #if self.debugMode:
        #    print(">> Hidden Weiths After Update(training...) <<\n", self.hidden_weights)



    # 신경망에 질의
    def query(self, input_list, showResult=True):
        return self.forward(input_list, showResult)


    def forward(self, input_list, showResult=True):
        # 입력 list를 행렬로 변환
        inputs = np.array(input_list, ndmin=2)

        # 은닉층
        hidden_inputs = np.dot(inputs, self.hidden_weights)
        hidden_outputs = self.activation_function(hidden_inputs)

        # 출력층
        final_inputs = np.dot(hidden_outputs, self.final_weights)
        final_outputs = self.activation_function(final_inputs)

        if self.debugMode:
            print(">> final outputs <<\n", final_outputs)

        if showResult:
            index = np.argmax(final_outputs)
            print(">> result = %d [%f]" % (index, final_outputs[0, index]))

        return final_outputs


    # 입력값 정규화(0.01 ~ 1.00)
    def normalizeInputs(self, inputs, maxValue):
        return (np.asfarray(inputs) // maxValue * 0.99) + 0.01



"""
if __name__ == '__main__':
training_data = [1.0, 0.5, -1.5]
inputNodes = len(training_data)

training_label = [0.9, 0.5]
outputNodes = len(training_label)

hiddenNodes = 4

learningRate = 0.1

nn = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate, True)
nn.train(training_data, training_label)
nn.query(training_data)
"""