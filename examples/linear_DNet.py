import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from model import NNet
from layers import LinearLayer
from activations import ReLU, Sigmoid, LeakyReLU, Tanh
from loss import BinaryCrossEntropyLoss
from optimizers import SGD

# initialize the parameters of the dataset
n_samples = 10000
noise = 6
random_state = 1

# Create the dataset
x, y = make_classification(
    n_samples=n_samples, random_state=random_state
)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, random_state=1
)

# Initialize the model
model = NNet()

# Create the model structure
model.add(LinearLayer(x.shape[1], 20))
model.add(LeakyReLU())
 
model.add(LinearLayer(20,7))
model.add(LeakyReLU())
 
model.add(LinearLayer(7, 5))
model.add(LeakyReLU())
 
model.add(LinearLayer(5,1))
model.add(Sigmoid())

# set the loss functions and the optimize method
loss = BinaryCrossEntropyLoss()
optim = SGD(lr=0.0075)

# Train the model
costs = []

for epoch in range(7000):
    model.forward(x_train.T)
    cost = model.cost(y_train, loss)
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