"Model Class"

class NNet:
    """
    Model class.
    """
    def __init__(self):
        self.layers = []
        self.losses = []

    def add(self, layer):
        """
        Add layers and activations.
        """
        self.layers.append(layer)

    def forward(self, X):
        """
        Compute forward propagation.
        """
        for layer in self.layers:
            pred = layer.forward(X)
            X = pred

        self.pred = pred 
    
    def cost(self, Y, method):
        """
        Compute cost.
        """
        error = method.forward(self.pred, Y)
        self.grad = method.backward()

        return error 

    def backward(self):
        """
        Compute backward propagation.
        """
        for layer in reversed(self.layers):
            self.grad = layer.backward(self.grad)

    def optimize(self, method):
        """
        Optimize parameters.
        """
        for layer in reversed(self.layers):
            if layer.type == 'Linear': 
                layer.optimize(method)
