import numpy as np 

from .base import Base
from .activation import Softmax


class BCELoss(Base):
    """
    Class that implements the Binary Cross Entropy loss function.

    Given a target math:`y` and an estimate math:`\hat{y}` the 
    Binary Cross Entropy loss can be written as:

    .. math::
        \begin{aligned}
            l_n & = \left[ y_n \cdot \log \hat{y_n} + (1 - y_n) \cdot \log (1 - \hat{y_n}) \right], \\
            L(\hat{y}, y) = \frac{\sum_{i=1}^{N}l_i}{N},
        \end{aligned}

    References
    ----------
    .. [1] Wikipedia - Cross entropy:
       https://en.wikipedia.org/wiki/Cross_entropy
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

        sub1 = np.dot(self.Y, np.log(self.Y_hat + self.eps).T)
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
        neg = np.divide(self.Y, self.Y_hat + self.eps)
        pos = np.divide(1 - self.Y, 1 + self.eps - self.Y_hat)
        grad = - (neg - pos)

        return grad


class MSELoss(Base):
    """
    Class that implements the Mean Squared Error loss function.

    Given a target math:`y` and an estimate math:`\hat{y}` the 
    mean squared error can be written as:

    .. math::
        \begin{aligned}
            l_n & = (y_n - \hat{y_n})^2, \\
            L(\hat{y}, y) = \frac{\sum_{i=1}^{N}l_i}{N},
        \end{aligned}

    References
    ----------
    .. [1] Wikipedia - Mean squared error:
       https://en.wikipedia.org/wiki/Mean_squared_error
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

        se = np.power(self.Y - self.Y_hat, 2)
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
        grad = - 2 * (self.Y - self.Y_hat) 

        return grad


class MAELoss(Base):
    """
    Class that implements the Mean Absolute Error loss function.

    Given a target math:`y` and an estimate math:`\hat{y}` the 
    mean absolute error can be written as:

    .. math::
        \begin{aligned}
            l_n & = |y_n - \hat{y_n}|, \\
            L(\hat{y}, y) = \frac{\sum_{i=1}^{N}l_i}{N},
        \end{aligned}

    References
    ----------
    .. [1] Wikipedia - Mean absolute error:
       https://en.wikipedia.org/wiki/Mean_absolute_error
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

        ae = np.abs(self.Y - self.Y_hat)
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
        grad = - np.sign(self.Y - self.Y_hat)

        return grad


class CrossEntropyLoss(Base):
    """
    Class that implements the Cross entropy loss function.

    Given a target math:`y` and an estimate math:`\hat{y}` the 
    Cross Entropy loss can be written as:

    .. math::
        \begin{aligned}
            l_{\hat{y}, class} = -\log\left(\frac{\exp(\hat{y_n}[class])}{\sum_j \exp(\hat{y_n}[j])}\right), \\
            L(\hat{y}, y) = \frac{\sum^{N}_{i=1} l_{i, class[i]}}{\sum^{N}_{i=1} weight_{class[i]}},
        \end{aligned}

    References
    ----------
    .. [1] Wikipedia - Cross entropy:
       https://en.wikipedia.org/wiki/Cross_entropy    
    """
    def __init__(self):
        self.type = 'CELoss'
        self.eps = 1e-15
        self.softmax = Softmax()
    
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
        self.Y_hat = self.softmax.forward(Y_hat)

        _loss = - Y * np.log(self.Y_hat)
        loss = np.sum(_loss, axis=0).mean()

        return np.squeeze(loss) 

    def backward(self):
        """
        Computes the backward propagation.

        Returns
        -------
        grad : numpy.array
            Array containg the gradients of the weights.
        """
        grad = self.Y_hat - self.Y
        
        return grad


class NLLLoss(Base):
    """
    Class that implements the Negative log likelihood loss function.

    Given a target math:`y` and an estimate math:`\hat{y}` the 
    Negative log likelihood loss can be written as:

    .. math::
        \begin{aligned}
            l_{\hat{y}, class} = -\log\left(\frac{\exp(\hat{y_n}[class])}{\sum_j \exp(\hat{y_n}[j])}\right), \\
            L(\hat{y}, y) = \frac{\sum^{N}_{i=1} l_{i, class[i]}}{\sum^{N}_{i=1} weight_{class[i]}},
        \end{aligned}

    References
    ----------
    .. [1] Pytorch - NLLLOSS:
       https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html   
    """
    def __init__(self):
        self.type = 'NLLLoss'
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

        _loss = - Y * np.log(self.Y_hat)
        loss = np.sum(_loss, axis=0).mean()

        return np.squeeze(loss) 

    def backward(self):
        """
        Computes the backward propagation.

        Returns
        -------
        grad : numpy.array
            Array containg the gradients of the weights.
        """
        grad = self.Y_hat - self.Y
        
        return grad