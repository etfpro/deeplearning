from mnist import load_mnist
from PIL import Image
import numpy as np

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()



(train_data, train_label), (test_data, test_label) = load_mnist(flatten=True, normalize=False)

img = train_data[7]
label = train_label[7]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)