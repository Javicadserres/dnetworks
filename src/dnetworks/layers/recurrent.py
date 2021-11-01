import numpy as np
from .base import RNNBase
from .linear import LinearLayer
from .activation import Tanh, Softmax, Sigmoid


class RNNCell(RNNBase):
    """
    Recurrent neural network cell implementation.
    """
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the parameters with the input and hidden
        dimensions. 

        Parameters
        ----------
        input_dim : int
            Dimension of the input. 
        hidden_dim : int
            Number of units in the RNN cell.
        """
        super(RNNCell, self).__init__(input_dim, hidden_dim)

        self.lineal = LinearLayer(input_dim + hidden_dim, hidden_dim)
        self.tanh = Tanh()


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
        hidden : numpy.array
            Array containing the output of the hidden state.
        """
        if hidden is None: hidden = np.zeros((self.hidden_dim, 1))

        # combine the input with the hidden state
        self.combined = np.concatenate((input_X, hidden), axis=0)
        input_hidden = self.lineal.forward(self.combined)
        # hidden state
        hidden = self.tanh.forward(input_hidden)

        return hidden    


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
        # derivative of the hyperbolic tangent
        self.tanh.A = hidden
        d_tanh = self.tanh.backward(dZ)

        # derivative of the hidden state
        parameters = self._retrieve_parameters(self.lineal, d_hidden)
        self.lineal.A = combined
        d_hidden = self.lineal.backward(d_tanh)
        self.lineal = self._update_parameters(self.lineal, parameters)

        return d_hidden[-self.hidden_dim:, :]


    def optimize(self, method):
        """
        Updates the parameters of the model using a given optimize 
        method.

        Parameters
        ----------
        method: Class
            Method to use in order to optimize the parameters.
        """
        self.lineal.optimize(method)


class RNN(RNNBase):
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
        super(RNN, self).__init__(input_dim, hidden_dim)
        self.output_dim = output_dim
        self.type = 'Recurrent'

        self.rnn_cell = RNNCell(input_dim, hidden_dim)
        self.lineal = LinearLayer(hidden_dim, output_dim)
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
        self.input_X = input_X

        if hidden is None: hidden = np.zeros((self.hidden_dim, 1))

        self.hiddens = [hidden]
        self.combines = []
        outputs = []

        for input in input_X:
            # compute the output
            hidden = self.rnn_cell.forward(input, hidden)
            input_softmax = self.lineal.forward(hidden)
            output = self.softmax.forward(input_softmax)

            # save the outputs for the backward prop
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
        reverse = zip(
            reversed(dZ), reversed(self.hiddens), reversed(self.combines)
        )

        for dz, hidden, combined in reverse:
            # derivative of the output
            parameters = self._retrieve_parameters(self.lineal, d_hidden)
            self.lineal.A = hidden
            d_output = self.lineal.backward(dz) + d_hidden
            self.lineal = self._update_parameters(self.lineal, parameters)

            # derivative of the hidden state
            d_hidden = self.rnn_cell.backward(
                d_output, d_hidden, hidden, combined
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
        self.lineal.optimize(method)
        self.rnn_cell.optimize(method)


class LSTMCell(RNNBase):
    """
    Recurrent neural network cell implementation.
    """
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the parameters with the input and hidden
        dimensions. 

        Parameters
        ----------
        input_dim : int
            Dimension of the input. 
        hidden_dim : int
            Number of units in the RNN cell.
        """
        super(LSTMCell, self).__init__(input_dim, hidden_dim)

        self.lineal_f = LinearLayer(input_dim + hidden_dim, hidden_dim)
        self.sigmoid_f = Sigmoid()

        self.lineal_i = LinearLayer(input_dim + hidden_dim, hidden_dim)
        self.sigmoid_i = Sigmoid()

        self.lineal_tanh = LinearLayer(input_dim + hidden_dim, hidden_dim)
        self.tanh_i = Tanh()

        self.lineal_o = LinearLayer(input_dim + hidden_dim, hidden_dim)
        self.sigmoid_o = Sigmoid()
 
        self.tanh_c = Tanh()


    def forward(self, input_X, hidden=None, cell_state=None):
        """
        Computes the forward propagation of the RNN.

        Parameters
        ----------
        input_X : numpy.array or list
            List containing all the inputs that will be used to 
            propagete along the RNN cell.

        Returns
        -------
        hidden : numpy.array
            Array containing the output of the hidden state.
        """
        if hidden is None: hidden = np.zeros((self.hidden_dim, 1))

        # combine the input with the hidden state
        self.combined = np.concatenate((input_X, hidden), axis=0)

        forget_input = self.lineal_f.forward(self.combined)
        forget_layer = self.sigmoid_f.forward(forget_input)
        
        input_input = self.lineal_i.forward(self.combined)
        input_layer = self.sigmoid_i.forward(input_input)

        tanh_input = self.lineal_tanh.forward(self.combined)
        tanh_layer = self.tanh_i.forward(tanh_input)

        output_input = self.lineal_o.forward(self.combined)
        output_layer = self.sigmoid_o.forward(output_input)

        cell_state = forget_layer * cell_state + input_layer * tanh_layer
        _tanh_c = self.tanh_c.forward(cell_state)
        hidden = output_layer * _tanh_c

        return hidden, cell_state


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
        # derivative of the hyperbolic tangent
        self.tanh.A = hidden
        d_tanh_c = self.tanh_c.backward(dZ)

        # derivative of the hidden state
        parameters = self._retrieve_parameters(self.lineal, d_hidden)
        self.lineal.A = combined
        d_hidden = self.lineal.backward(d_tanh_c)
        self.lineal = self._update_parameters(self.lineal, parameters)

        return d_hidden[-self.hidden_dim:, :]


    def optimize(self, method):
        """
        Updates the parameters of the model using a given optimize 
        method.

        Parameters
        ----------
        method: Class
            Method to use in order to optimize the parameters.
        """
        self.lineal.optimize(method)