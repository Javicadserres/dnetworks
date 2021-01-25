import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from model import NNet
from layers.layers import LinearLayer
from activations import ReLU
from loss import CrossEntropyLoss
from optimizers import Adam
from layers.convent import Conv2D
from layers.pooling_layers import MaxPooling2D, AveragePooling2D
from layers.flatten_layer import Flatten

digits = load_digits()
images = digits.images
m, h, w = images.shape
images = images.reshape(m, 1, h, w)

target = digits.target

def one_hot_encoding(Y):
    """
    One hot enconding method.
    """
    one_hot = np.zeros((Y.size, Y.max() + 1))
    one_hot[np.arange(Y.size), Y] = 1

    return one_hot

x_train, x_test, y_train, y_test = train_test_split(
    images, target, test_size=0.4, random_state=1
)
y_train = one_hot_encoding(y_train)

# Initialize the model
model = NNet()

# Create the model structure
model.add(Conv2D(1, 2, kernel_size=(2, 2), stride=2, padding=1))
model.add(AveragePooling2D(kernel_size=(2, 2), stride=1, padding=1))
model.add(ReLU())

model.add(Conv2D(2, 1, kernel_size=(2, 2), stride=1, padding=0))
model.add(AveragePooling2D(kernel_size=(2, 2), stride=1, padding=0))
model.add(ReLU())

model.add(Flatten())
model.add(LinearLayer(16, 10))

# set the loss functions and the optimize method
loss = CrossEntropyLoss()
optim = Adam(lr=0.05)

# Train the model
costs = []

for epoch in range(10000):
    model.forward(x_train.T)
    cost = model.cost(y_train.T, loss)
    model.backward()
    model.optimize(optim)

    if epoch % 100 == 0:
        print ("Cost after iteration %epoch: %f" %(epoch, cost))
        costs.append(cost)

# plot the loss evolution
plt.plot(np.squeeze(costs)[1:])
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.show()