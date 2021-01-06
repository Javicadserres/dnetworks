import numpy as np 


class BinaryCrossEntropyLoss:
    def __init__(self):
        self.type = 'BCELoss'
    
    def forward(self, Y_hat, Y):
        """
        Forward BCELoss
        """
        self.Y = Y
        self.Y_hat = Y_hat

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
        self.type = 'MSELoss'
    
    def forward(self, Y_hat, Y):
        """
        Forward MSELoss
        """
        self.Y = Y
        self.Y_hat = Y_hat

        se = np.power(self.Y_hat - self.Y, 2)
        mse = np.mean(se)

        return np.squeeze(mse)

    def backward(self):
        """
        Backward MSELoss
        """
        grad = np.mean(2 * (self.Y_hat - self.Y))

        return grad


class MAELoss:
    def __init__(self):
        self.type = 'MAELoss'
    
    def forward(self, Y_hat, Y):
        """
        Forward MAELoss
        """
        self.Y = Y
        self.Y_hat = Y_hat

        ae = np.abs(self.Y_hat - self.Y)
        mae = np.mean(ae)

        return np.squeeze(mae)

    def backward(self):
        """
        Backward MAELoss
        """
        grad = np.sign(self.Y_hat - self.Y)

        return grad


class CrossEntropyLoss:
    def __init__(self):
        self.type = 'CELoss'
    
    def forward(self, Y_hat, Y):
        """
        Forward CELoss
        """
        self.Y = Y
        self.Y_hat = Y_hat

        pass

    def backward(self):
        """
        Backward CELoss
        """
        pass
