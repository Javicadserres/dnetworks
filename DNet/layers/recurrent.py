import numpy as np
from .base import Base
from .linear import LinearLayer
from .activation import Tanh, Softmax


class RNNCell(Base):
    """
    Recurrent neural network cell implementation.
    """
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Initialize the parameters with the input, output and hidden
        dimensions. 

        Parameters
        ----------
        input_dim : int
            Dimension of the input. 
        output_dim : int
            Dimension of the output.
        hidden_dim : int
            Number of units in the RNN cell.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.lineal_h = LinearLayer(input_dim + hidden_dim, hidden_dim)
        self.lineal_o = LinearLayer(hidden_dim, output_dim)
        self.tanh = Tanh()
        self.softmax = Softmax()


    def forward(self, input_X, hidden=None):
        """
        Computes the forward propagation of the RNN.

        Parameters
        ----------
        input_X : numpy.array or list
            List containing all the inputs that will be used to 
            propagete along the RNN cell.

        Returns
        -------
        y_preds : list
            List containing all the preditions for each input of the
            input_X list.
        """
        if hidden is None: hidden = np.zeros((self.hidden_dim, 1))

        # combine the input with the hidden state
        self.combined = np.concatenate((input_X, hidden), axis=0)
        input_hidden = self.lineal_h.forward(self.combined)
        # hidden state
        hidden = self.tanh.forward(input_hidden)
        # output
        input_softmax = self.lineal_o.forward(hidden)
        output = self.softmax.forward(input_softmax)

        return output, hidden    


    def backward(self, dZ, d_hidden=0, hidden=None, combined=None):  
        """
        Computes the backward propagation of the model.

        Parameters
        ----------
        dZ : numpy.array
            The gradient of the of the output with respect to the
            next layer.

        Returns
        -------
        d_hidden : numpy.array
            The gradient of the input with respect to the current 
            layer.
        """  
        # derivative of the output
        parameters = self._retrieve_parameters(self.lineal_o, d_hidden)
        self.lineal_o.A = hidden
        d_output = self.lineal_o.backward(dZ) + d_hidden
        self.lineal_o = self._update_parameters(
            self.lineal_o, parameters
        )

        # derivative of the hyperbolic tangent
        self.tanh.A = hidden
        d_tanh = self.tanh.backward(d_output)

        # derivative of the hidden state
        parameters = self._retrieve_parameters(self.lineal_h, d_hidden)
        self.lineal_h.A = combined
        d_hidden = self.lineal_h.backward(d_tanh)
        self.lineal_h = self._update_parameters(
            self.lineal_h, parameters
        )
        d_hidden = d_hidden[-self.hidden_dim:, :]

        return d_hidden


    def optimize(self, method):
        """
        Updates the parameters of the model using a given optimize 
        method.

        Parameters
        ----------
        method: Class
            Method to use in order to optimize the parameters.
        """
        for layer in [self.lineal_o, self.lineal_h]:
            layer.optimize(method)


    def _retrieve_parameters(self, layer, d_hidden):
        """
        """
        if isinstance(d_hidden, int):
            parameters = [0, 0]
        else:
            parameters = [layer.dW, layer.db]

        return parameters


    def _update_parameters(self, layer, parameters):
        """
        Updates parameters
        """
        # actualize parameters
        layer.dW += parameters[0]
        layer.db += parameters[1]

        return layer


class RNN(Base):
    """
    Implementation of a Recurrent neural network.
    """
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Initialize the parameters with the input, output and hidden
        dimensions. 

        Parameters
        ----------
        input_dim : int
            Dimension of the input. 
        output_dim : int
            Dimension of the output.
        hidden_dim : int
            Number of units in the RNN cell.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.type = 'Recurrent'

        self.rnn_cell = RNNCell(input_dim, output_dim, hidden_dim)


    def forward(self, input_X, hidden=None):
        """
        Computes the forward propagation of the RNN.

        Parameters
        ----------
        input_X : numpy.array or list
            List containing all the inputs that will be used to 
            propagete along the RNN cell.

        Returns
        -------
        y_preds : list
            List containing all the preditions for each input of the
            input_X list.
        """
        self.input_X = input_X

        if hidden is None: hidden = np.zeros((self.hidden_dim, 1))

        self.hiddens = [hidden]
        self.combines = []
        outputs = []

        for input in input_X:
            output, hidden = self.rnn_cell.forward(input, hidden)
            outputs.append(output.tolist())
            self.hiddens.append(hidden)
            self.combines.append(self.rnn_cell.combined)

        return np.array(outputs)


    def backward(self, dZ, d_hidden=0):  
        """
        Computes the backward propagation of the model.

        Parameters
        ----------
        dZ : numpy.array
            The gradient of the of the output with respect to the
            next layer.

        Returns
        -------
        d_hidden : numpy.array
            The gradient of the input with respect to the current 
            layer.
        """
        d_hidden = 0
        reverse = zip(
            reversed(dZ), reversed(self.hiddens), reversed(self.combines)
        )

        for dz, hidden, combined in reverse:
            d_hidden = self.rnn_cell.backward(
                dz, d_hidden, hidden, combined
            )

        return d_hidden


    def optimize(self, method):
        """
        Updates the parameters of the model using a given optimize 
        method.

        Parameters
        ----------
        method: Class
            Method to use in order to optimize the parameters.
        """
        self.rnn_cell.optimize(method)