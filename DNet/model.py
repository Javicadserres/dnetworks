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
        loss = method(self.pred, Y)
        error = loss.forward()
        self.grad = loss.backward()

        return error 

    def backward(self):
        """
        Compute backward propagation.
        """
        for layer in reversed(self.layers):
            self.grad = layer.backward(self.grad)

    def optimize(self):
        """
        Optimize parameters.
        """
        for layer in reversed(self.layers):
            if layer.type == 'Linear': 
                layer.optimize()
