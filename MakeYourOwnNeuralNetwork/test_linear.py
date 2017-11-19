import numpy as np
import neuralNetwork as nn


inputs = [[0, 0, 0],
          [1, 0, 1],
          [0, 1, 1],
          [1, 1, 1]]


inputNodes = 2
outputNodes = 1
learningRate = 0.1

network = nn.neuralNetwork(inputNodes, outputNodes, learningRate)

epochs = 10000

for e in range(epochs):
    for input in inputs:
        # 레이블(실제값은 0.99, 나머지는 0.01)
        label = max(0.01, input[-1] - 0.01)
        network.train(network.normalizeInputs(input[:-1], 1.0), label)

epochs = 10
for e in range(epochs):
    print()
    for input in inputs:
        network.forward(input[:-1])