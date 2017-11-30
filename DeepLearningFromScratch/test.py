from mnist import load_mnist
from PIL import Image
import numpy as np

class MulLayer:
    def __init(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        # 미분 역전파 시 활용을 위해 순전파 동안 입력값(이전 계층의 출력값)들을 저장
        self.x = x
        self.y = y

        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        # 덧셈은 미분에 영향을 미치지 않기 때문에 입력값들 저장 불필요
        return x + y

    def backward(self, dout):
        return dout, dout



apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)

# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)
