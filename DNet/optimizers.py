import numpy as np


class SGD:
    """
    Class that implements the gradient descent algorithm.
    """
    def __init__(self, lr=0.0075, beta=0.9):
        """
        Parameters
        ----------
        lr : int
            Learing rate to use for the gradient descent.
        beta : int
            Beta parameter.
        """
        self.beta = beta
        self.lr = lr

    def optim(self, weigths, bias, dW, db, velocities=None):
        """
        Parameters
        ---------
        weights : numpy.array
            Weigths of a given layer.
        bias : numpy.array
            Bias of a given layer.
        dW : numpy.array
            The gradients of the weights.
        db : numpy.array
            The gradients of the bias
        velocities : tuple
            Tuple containing the velocities to compute the gradient
            descent with momentum.

        Returns
        -------
        weights : numpy.array
            Updated weigths of the given layer.
        bias : numpy.array
            Updated bias of the given layer.
        (V_dW, V_db) : tuple
            Tuple of ints containing the velocities for the weights
            and biases.
        """
        if velocities is None: velocities = (0, 0)

        V_dW, V_db = self._update_velocities(
            dW, db, self.beta, velocities
        )

        weigths -= self.lr * V_dW
        bias -= self.lr * V_db

        return weigths, bias, (V_dW, V_db)

    def _update_velocities(self, dW, db, beta, velocities):
        """
        Updates the velocities of the derivates of the weights and 
        bias.
        """
        V_dW, V_db = velocities

        V_dW = beta * V_dW + (1 - beta) * dW
        V_db = beta * V_db + (1 - beta) * db

        return V_dW, V_db


class RMSprop:
    """
    Class that implements the RMSprop algorithm.
    """
    def __init__(self, lr=0.0075, beta=0.9):
        """
        Parameters
        ----------
        lr : int
            Learing rate to use for the gradient descent.
        beta : int
            Beta parameter to compute the sqaures.
        """
        self.beta = beta
        self.lr = lr

    def optim(self, weigths, bias, dW, db, squares=None):
        """
        Parameters
        ---------
        weights : numpy.array
            Weigths of a given layer.
        bias : numpy.array
            Bias of a given layer.
        dW : numpy.array
            The gradients of the weights.
        db : numpy.array
            The gradients of the bias
        squares : tuple
            Tuple containing the square to compute the gradient
            descent with momentum.

        Returns
        -------
        weights : numpy.array
            Updated weigths of the given layer.
        bias : numpy.array
            Updated bias of the given layer.
        (S_dW, S_db) : tuple
            Tuple containing the square to compute the gradient
            descent with momentum.
        """
        if squares is None: squares = (0, 0)

        S_dW, S_db = self._update_squares(
            dW, db, self.beta, squares
        )

        weigths -= self.lr * dW / np.sqrt(S_dW)
        bias -= self.lr * db / np.sqrt(S_db)

        return weigths, bias, (S_dW, S_db)

    def _update_squares(self, dW, db, beta, squares):
        """
        Updates the squares of the derivates of the weights and bias.
        """
        S_dW, S_db = squares

        S_dW = beta * S_dW + (1 - beta) * np.power(dW, 2)
        S_db = beta * S_db + (1 - beta) * np.power(db, 2)

        return S_dW, S_db


class Adam:
    """
    Class that implements the Adaptive moment stimation algorithm.
    """
    def __init__(self, lr=0.0075, betas=(0.9, 0.99), epsilon=1e-18):
        """
        Parameters
        ----------
        lr : int
            Learing rate to use for the gradient descent.
        betas : tuple of int
            Betas parameters.
        epsilon : int
        """
        self.betas = betas
        self.lr = lr
        self.epsilon = epsilon

    def optim(self, weigths, bias, dW, db, vel_square=None):
        """
        Parameters
        ---------
        weights : numpy.array
            Weigths of a given layer.
        bias : numpy.array
            Bias of a given layer.
        dW : numpy.array
            The gradients of the weights.
        db : numpy.array
            The gradients of the bias
        vel_square : tuple
            Tuple containing the square to compute the gradient
            descent with momentum.

        Returns
        -------
        weights : numpy.array
            Updated weigths of the given layer.
        bias : numpy.array
            Updated bias of the given layer.
        vel_square : tuple
            Tuple of ints containing:
            1. The velocities for the weights and biases.
            2. The squares for the weights and biases.
            3. The current epoch.
        """
        if vel_square is None:
            V_dW, V_db, S_dW, S_db, epoch = (0, 0, 0, 0, 1)
        else:
            V_dW, V_db, S_dW, S_db, epoch = vel_square
        
        beta1, beta2 = self.betas

        # update and correct velocities
        V_dW, V_db = self._update_velocities(
            dW, db, beta1, (V_dW, V_db)
        )
        V_dW_c, V_db_c = self._correct_update(V_dW, V_db, beta1, epoch)

        # update and correct squares
        S_dW, S_db = self._update_squares(
            dW, db, beta2, (S_dW, S_db)
        )
        S_dW_c, S_db_c = self._correct_update(S_dW, S_db, beta2, epoch)

        # update parameters
        weigths -= self.lr * V_dW_c / (np.sqrt(S_dW_c) + self.epsilon)
        bias -= self.lr * V_db_c / (np.sqrt(S_db_c) + self.epsilon)

        epoch += 1
        vel_square = (V_dW, V_db, S_dW, S_db, epoch)

        return weigths, bias, vel_square

    def _update_velocities(self, dW, db, beta, velocities):
        """
        Updates the velocities of the derivates of the weights and 
        bias.
        """
        V_dW, V_db = velocities

        V_dW = beta * V_dW + (1 - beta) * dW
        V_db = beta * V_db + (1 - beta) * db

        return V_dW, V_db

    def _update_squares(self, dW, db, beta, squares):
        """
        Updates the squares of the derivates of the weights and bias.
        """
        S_dW, S_db = squares

        S_dW = beta * S_dW + (1 - beta) * np.power(dW, 2)
        S_db = beta * S_db + (1 - beta) * np.power(db, 2)

        return S_dW, S_db

    def _correct_update(self, uptdate_w, update_b, beta, epoch):
        """
        Corrects the updates made for the velocities and squares of 
        the derivatives of the weights and bias.
        """
        cor_update_w = uptdate_w / (1 - np.power(beta, epoch))
        cor_update_b = update_b / (1 - np.power(beta, epoch))

        return cor_update_w, cor_update_b