import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from model import NNet
from layers import LinearLayer
from activations import ReLU, Sigmoid, LeakyReLU, Tanh
from loss import BinaryCrossEntropyLoss
from optimizers import SGD, RMSprop, Adam

print('hello')