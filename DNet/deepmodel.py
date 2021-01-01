import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils import initialize_parameters_deep
from linear_layer import L_model_forward
from loss_functions import compute_loss 
from backwards.backward import L_model_backward
from optimizers import optimize


def L_layer_model(
    X, 
    Y, 
    layers_dims, 
    lr = 0.0075, 
    num_iterations = 7000, 
    print_cost=False
):
    """
    Implements a L-layer neural network.
    
    Parameters
    ----------
    X : np.array
        Data containing the variables used to predict the labels.
    Y : np.array
        Data containing the labels.
    layers_dims : list
        List containing the input size and each layer size, of length 
        (number of layers + 1). The first values is the number
        of input variables into de model.
    lr : int
        Learning rate of the optimization function.
    num_iterations : int
        Number of iterations of the optimization loop.
    print_cost : bool
        If True, it prints the cost every 100 steps.
    
    Returns
    -------
    parameters : dict
        Dictionary containing the parameters learnt by the model.
    costs : list
        List containing the costs of every 100 iterations.
    """
    np.random.seed(1)
    costs = []
    
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation
        Y_hat, caches = L_model_forward(X, parameters)
        # Compute cost.
        cost = compute_loss(Y_hat, Y, cost_function='BCE')
        # Backward propagation.
        grads = L_model_backward(Y_hat, Y, caches)
        # Update parameters.
        parameters = optimize(parameters, grads, lr)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
    
    return parameters, costs


# initialize the parameters of the dataset
n_samples = 10000
noise = 6
random_state = 1

x, y = make_classification(
    n_samples=n_samples, random_state=random_state
)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, random_state=1
)

layer_dims = [x_train.shape[1], 20, 7, 5, 1]

parameters, costs = L_layer_model(
    x_train.T, y_train , layer_dims, print_cost=True
)

plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.show()