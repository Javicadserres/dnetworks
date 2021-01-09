import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from model import NNet
from layers import LinearLayer
from activations import ReLU, Sigmoid, LeakyReLU, Tanh, Softmax
from loss import BinaryCrossEntropyLoss, CrossEntropyLoss
from optimizers import SGD, RMSprop, Adam

# initialize the parameters of the dataset
n_samples = 10000
n_classes = 5
rand = 1

# Create the dataset
x, y = make_classification(
    n_samples=n_samples, 
    random_state=rand, 
    n_classes=n_classes,
    n_informative=7
)

def one_hot_encoding(Y):
    """
    One hot enconding method.
    """
    one_hot = np.zeros((Y.size, Y.max() + 1))
    one_hot[np.arange(Y.size), Y] = 1

    return one_hot

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, random_state=rand
)

y_train = one_hot_encoding(y_train)

# Initialize the model
model = NNet()

# Create the model structure
model.add(LinearLayer(x.shape[1], 20))
model.add(Sigmoid())
 
model.add(LinearLayer(20, 7))
model.add(Sigmoid())
 
model.add(LinearLayer(7, 5))
model.add(Sigmoid())
 
model.add(LinearLayer(5, 5))
model.add(Softmax())

# set the loss functions and the optimize method
loss = CrossEntropyLoss()
optim = Adam()

# Train the model
costs = []

for epoch in range(7000):
    model.forward(x_train.T)
    cost = model.cost(y_train.T, loss)
    model.backward()
    model.optimize(optim)

    if epoch % 100 == 0:
        print ("Cost after iteration %epoch: %f" %(epoch, cost))
        costs.append(cost)

# plot the loss evolution
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.show()