import numpy as np 


class BinaryCrossEntropyLoss:
    """
    Class that implements the Binary Cross Entropy loss function.
    """
    def __init__(self):
        self.type = 'BCELoss'
        self.eps = 1e-15
    
    def forward(self, Y_hat, Y):
        """
        Computes the forward propagation.

        Parameters
        ----------
        Y_hat : numpy.array
            Array containing the predictions.
        Y : numpy.array
            Array with the real labels.
        
        Returns
        -------
        Numpy.arry containing the cost.
        """
        self.Y = Y
        self.Y_hat = Y_hat

        m = len(self.Y)

        sub1 = np.dot(self.Y, np.log(self.Y_hat).T)
        sub2 = np.dot(1 - self.Y, np.log(1 + self.eps - self.Y_hat).T)

        loss = - (1./m) * (sub1 + sub2)

        return np.squeeze(loss)

    def backward(self):
        """
        Computes the backward propagation.

        Returns
        -------
        grad : numpy.array
            Array containg the gradients of the weights.
        """
        neg = np.divide(self.Y, self.Y_hat)
        pos = np.divide(1 - self.Y, 1 + self.eps - self.Y_hat)
        grad = - (neg - pos)

        return grad


class MSELoss:
    """
    Class that implements the Mean Squared Error loss function.
    """
    def __init__(self):
        self.type = 'MSELoss'
    
    def forward(self, Y_hat, Y):
        """
        Computes the forward propagation.

        Parameters
        ----------
        Y_hat : numpy.array
            Array containing the predictions.
        Y : numpy.array
            Array with the real labels.
        
        Returns
        -------
        Numpy.arry containing the cost.
        """
        self.Y = Y
        self.Y_hat = Y_hat

        se = np.power(self.Y_hat - self.Y, 2)
        mse = np.mean(se)

        return np.squeeze(mse)

    def backward(self):
        """
        Computes the backward propagation.

        Returns
        -------
        grad : numpy.array
            Array containg the gradients of the weights.
        """
        grad = 2 * (self.Y_hat - self.Y)

        return grad


class MAELoss:
    """
    Class that implements the Mean Absolute Error loss function.
    """
    def __init__(self):
        self.type = 'MAELoss'
    
    def forward(self, Y_hat, Y):
        """
        Computes the forward propagation.

        Parameters
        ----------
        Y_hat : numpy.array
            Array containing the predictions.
        Y : numpy.array
            Array with the real labels.
        
        Returns
        -------
        Numpy.arry containing the cost.
        """
        self.Y = Y
        self.Y_hat = Y_hat

        ae = np.abs(self.Y_hat - self.Y)
        mae = np.mean(ae)

        return np.squeeze(mae)

    def backward(self):
        """
        Computes the backward propagation.

        Returns
        -------
        grad : numpy.array
            Array containg the gradients of the weights.
        """
        grad = np.sign(self.Y_hat - self.Y)

        return grad


class CrossEntropyLoss:
    """
    Class that implements the Mean Absolute Error loss function.
    """
    def __init__(self):
        self.type = 'CELoss'
        self.eps = 1e-15
    
    def forward(self, Y_hat, Y):
        """
        Computes the forward propagation.

        Parameters
        ----------
        Y_hat : numpy.array
            Array containing the predictions.
        Y : numpy.array
            Array with the real labels.
        
        Returns
        -------
        Numpy.arry containing the cost.
        """
        self.Y = Y
        self.Y_hat = Y_hat

        loss = - np.sum(Y * np.log(Y_hat + self.eps), axis=0)
        cost = np.mean(loss)

        return np.squeeze(cost) 

    def backward(self):
        """
        Computes the backward propagation.

        Returns
        -------
        grad : numpy.array
            Array containg the gradients of the weights.
        """
        grad = np.subtract(self.Y_hat, self.Y)
        
        return grad
