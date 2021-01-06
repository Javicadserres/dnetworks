"Model Class"

class NNet:
    """
    Class containing the structure to create a model.
    """
    def __init__(self):
        """
        Initialize the layers and losses as an empty list.
        """
        self.layers = []
        self.losses = []

    def add(self, layer):
        """
        Adds layers or activations to the model.

        Parameters
        ----------
        layer : DNet.layers or DNet.activations
            The layer and/or activation to add in the model.
        
        Example
        -------
        >>> from DNet.model import NNet
        >>> from DNet.activations import LeakyReLU
        >>> from DNet.layers import LinearLayer

        >>> model = NNet()
        >>> model.add(LinearLayer(20,7))
        >>> model.add(LeakyReLU())
        """
        self.layers.append(layer)

    def forward(self, X):
        """
        Computes the forward propagation.

        Parameters
        ----------
        X : numpy.array
            Array containing the X variables to use in the model.
        
        Returns
        -------
        pred : numpy.array
            Predictions made by the model for a given set of weights.
        """
        for layer in self.layers:
            pred = layer.forward(X)
            X = pred

        self.pred = pred

        return pred 
    
    def cost(self, Y, method):
        """
        Compute the cost using a given loss method.

        Parameters
        ----------
        Y : numpy.array
            The real labels to predict.
        method : DNet.loss 
            Loss method to use.
        
        Returns
        -------
        error : numpy.array
            The error of the model.

        Examples
        --------
        >>> from DNet.loss import BinaryCrossEntropyLoss

        >>> loss = BinaryCrossEntropyLoss()
        >>> cost = model.cost(y_train, loss)
        """
        error = method.forward(self.pred, Y)
        self.grad = method.backward()

        return error 

    def backward(self):
        """
        Computes the backward propagation of the model.
        """
        for layer in reversed(self.layers):
            self.grad = layer.backward(self.grad)

    def optimize(self, method):
        """
        Updates the parameters of the model using a given optimize
        method.

        Parameters
        ----------
        method : DNet.optimizers
            Optimization method to use in order to obtain the parameters.
        
        Examples
        --------
        >>> from DNet.optimizers import SGD
        
        >>> optim = SGD(lr=0.0075)
        >>> model.optimize(optim)
        """
        for layer in reversed(self.layers):
            if layer.type == 'Linear': 
                layer.optimize(method)
