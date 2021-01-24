import numpy as np
from convent import Conv2D
from pooling_layers import MaxPooling2D, AveragePooling2D

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    
    ### START CODE HERE ### (≈ 1 line)
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values = (0,0))
    ### END CODE HERE ###
    
    return X_pad


# GRADED FUNCTION: conv_single_step

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = a_slice_prev * W
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(b)
    ### END CODE HERE ###

    return Z


# GRADED FUNCTION: conv_forward

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    # Compute the dimensions of the CONV output volume using the formula given above. 
    # Hint: use int() to apply the 'floor' operation. (≈2 lines)
    n_H = int(((n_H_prev + (2 * pad) - f) / stride) + 1)
    n_W = int(((n_W_prev + (2 * pad) - f) / stride) + 1)

    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i, :, :, :]               # Select ith training example's padded activation
        
        for h in range(n_H):           # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            vert_start = stride * h
            vert_end = stride * h + f
            
            for w in range(n_W):       # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice" (≈2 lines)
                horiz_start = stride * w
                horiz_end = stride * w + f
                
                for c in range(n_C):   # loop over channels (= #filters) of the output volume
                                        
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
                                        
    ### END CODE HERE ###
    
    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache


# GRADED FUNCTION: pool_forward

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              
    
    ### START CODE HERE ###
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            vert_start = stride * h
            vert_end = stride * h + f
            
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                # Find the vertical start and end of the current "slice" (≈2 lines)
                horiz_start = stride * w
                horiz_end = stride * w + f
                
                for c in range (n_C):            # loop over the channels of the output volume
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    # Compute the pooling operation on the slice. 
                    # Use an if statement to differentiate the modes. 
                    # Use np.max and np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    ### END CODE HERE ###
    
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache


def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    
    ### START CODE HERE ###
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):                       # loop over the training examples
        
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]
        
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = stride * h 
                    vert_end = stride * h + f
                    horiz_start = stride * w
                    horiz_end = stride * w + f
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
                    
        # Set the ith training example's dA_prev to the unpadded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        if pad==0:
            dA_prev[i, :, :, :] = da_prev_pad
        else:
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    ### END CODE HERE ###
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db


def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    
    ### START CODE HERE ### (≈1 line)
    mask = x==np.max(x)
    ### END CODE HERE ###
    
    return mask


def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from shape (≈1 line)
    (n_H, n_W) = shape
    
    # Compute the value to distribute on the matrix (≈1 line)
    average = dz/(n_H * n_W)
    
    # Create a matrix where every entry is the "average" value (≈1 line)
    a = np.ones((n_H, n_W)) * average
    ### END CODE HERE ###
    
    return a


def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    
    ### START CODE HERE ###
    
    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):                       # loop over the training examples
        
        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i]
        
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        
                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask,dA[i,h,w,c])
                        
                    elif mode == "average":
                        
                        # Get the value a from dA (≈1 line)
                        da = dA[i,h,w,c]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = f,f
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da,shape)
                        
    ### END CODE ###
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev







class Pool():
    ''' Pooling layer
    '''

    def __init__(self, func='max', filter_dim=(2, 2), stride=2):
        '''
        Initialize the pooling layer
        :param func: max,mean or sum - max is set as default
        :param filter_dim: pooling size
        :param stride: stride size
        '''
        self.func = func
        self.filter_dim = filter_dim
        self.stride = stride
        self.params = []

    def forward(self, X):
        '''
        calculates the activation map of our pooling layer
        :param X: input
        :return: activation map
        '''
        # reshapes the images to that the depth channel is 1
        n_x, d_x, h_x, w_x = X.shape
        x_reshaped = X.reshape(n_x * d_x, 1, h_x, w_x)

        # uses helper function to get a column matrix
        x_col = Helper().im2col_indices(x_reshaped, self.filter_dim[0], self.filter_dim[1], padding=0,
                                        stride=self.stride)
        # calculates the output
        if self.func == 'sum':
            out_col = np.sum(x_col, axis=0)
        elif self.func == 'mean':
            out_col = np.mean(x_col, axis=0)
        else:
            out_col = np.max(x_col, axis=0)

        # reshapes the column matrix to a image matrix
        out_size = int((h_x - self.filter_dim[0]) / self.stride + 1)
        #out = out_col.reshape(n_x, d_x, out_size, out_size)
        out = out_col.reshape(out_size, out_size, n_x, d_x)
        out = out.transpose(2, 3, 0, 1)

        # saves input and input col for future use
        self.x_col = x_col
        self.X = X
        return out

    def backward(self, dout):
        '''
        backward path of our pooling layers
        :param dout: gradient
        :return: dX, no gradient on pooling layers
        '''
        # create a column vector out of our gradient
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()

        # replace the values that are not equal to our gradient with 0
        temp_x_col = self.x_col.copy()
        temp_x_col[temp_x_col != dout_flat] = 0

        # since our matrix is still stretched out we need to revert it back to a image matrix
        n_x, d_x, h_x, w_x = self.X.shape
        dX = Helper().col2im_indices(temp_x_col, (n_x * d_x, 1, h_x, w_x), self.filter_dim[0], self.filter_dim[1],
                                     padding=0, stride=self.stride)
        dX = dX.reshape(self.X.shape)
        return dX, []


class Helper():
    
    def __init__(self):
        None

    def get_im2col_indices(self, x_shape, field_height, field_width, padding=1, stride=1):
        # First figure out what the size of the output should be
        N, C, H, W = x_shape
        out_size = (H + 2 * padding - field_height) / stride + 1
        out_size = int(out_size)

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_size), out_size)

        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_size), out_size)

        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

        return (k, i, j)

    def im2col_indices(self, x, field_height, field_width, padding=1, stride=1):
        '''
        Helper functions. Image matrix to column matrix
        :param x: input data
        :param field_height: kernel height
        :param field_width: kernel width
        :param padding: padding size
        :param stride: stride size
        :return: column matrixs
        '''
        # Zero-pad the input
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        k, i, j = self.get_im2col_indices(x.shape, field_height, field_width, padding,
                                          stride)

        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols

    def col2im_indices(self, cols, x_shape, field_height=3, field_width=3, padding=1,
                       stride=1):
        '''
        Helper function. Column matrix to Image matrix
        :param cols: colmn matrix of a input
        :param x_shape: the original input shape
        :param field_height: kernel height
        :param field_width: kernel width
        :param padding: padding size
        :param stride: stride size
        :return: image matrix
        '''
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = self.get_im2col_indices(x_shape, field_height, field_width, padding,
                                          stride)
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]





np.random.seed(1)

hparameters = {"pad" : 0, "stride": 2}

import time


### CONVOLUTION TEST

# forward
A = np.random.randn(10,4,5,7)
model = Conv2D(4, 2, kernel_size=(2, 2), stride=2, padding=0)
Z_class = model.forward(A)

A_prev = A.transpose(0, 2, 3, 1)
W = model.weights.transpose(2, 3, 1, 0)
b = np.zeros((1,1,1,8))
Z_func, cache_conv = conv_forward(A_prev, W, b, hparameters)

np.testing.assert_almost_equal(Z_class, Z_func.transpose(0, 3, 1, 2))

# backward
dA_class = model.backward(Z_class)
dA, dW, db = conv_backward(Z_func, cache_conv)

np.testing.assert_almost_equal(dA_class, dA.transpose(0, 3, 1, 2))

print('Convolution cool!')

### POOLING TEST

np.random.seed(1)
# forward
A = np.random.randn(2,2,5,5)
model = AveragePooling2D(kernel_size=(2, 2), stride=1, padding=0, padding_constant=0)
A_class = model.forward(A)

A_prev = A.transpose(0, 2, 3, 1)
hparameters = {"stride" : 1, "f": 2}
A_func, cache = pool_forward(A_prev, hparameters, mode='average')

np.testing.assert_almost_equal(A_class, A_func.transpose(0, 3, 1, 2))

# backward
dA_class = model.backward(A_class)
dA = pool_backward(A_func, cache, mode='average')

np.testing.assert_almost_equal(dA_class, dA.transpose(0, 3, 1, 2))

print('Pooling cool!')