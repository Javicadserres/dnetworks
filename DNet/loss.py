import numpy as np 


class BinaryCrossEntropyLoss:
    def __init__(self, Y_hat, Y):
        self.Y = Y
        self.Y_hat = Y_hat
    
    def forward(self):
        """
        Forward BCELoss
        """
        m = len(self.Y)

        sub1 = np.dot(self.Y, np.log(self.Y_hat).T)
        sub2 = np.dot(1 - self.Y, np.log(1 - self.Y_hat).T)

        loss = - (1./m) * (sub1 + sub2)

        return np.squeeze(loss)

    def backward(self):
        """
        Backward BCELoss
        """
        neg = np.divide(self.Y, self.Y_hat)
        pos = np.divide(1 - self.Y, 1 - self.Y_hat)
        grad = - (neg - pos)

        return grad


class MSELoss:
    def __init__(self):
        pass


class MAELoss:
    def __init__(self):
        pass