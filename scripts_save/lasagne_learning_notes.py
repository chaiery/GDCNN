# Taking notes

import numpy as np
import theano

import theano.tensor as T

# When creating input layer, should specify the shape of the input

# The input is a matrix of 100*50
# It represent a batch of 100 data points, each data point is a vector of length 50
l_in = lasagne.layers.InputLayer((100, 50)) 

# Note that we did not specify the nonlinearity of the hidden layer. 
# A layer with rectified linear units will be created by default.
l_hidden = lasagne.layers.DenseLayer(l_in, num_units=200)


l_out = lasagne.layers.DenseLayer(l_hidden, num_units=10,
                                   nonlinearity=T.nnet.softmax)


# Initialization of parameters
# Case 1: callable built-in function or customized function
l = lasagne.layers.DenseLayer(l_in, num_units=100,
                              W=lasagne.init.Normal(0.01))

# Case 2: Theano shared variable
W = theano.shared(np.random.normal(0, 0.01, (50, 100)))
l = lasagne.layers.DenseLayer(l_in, num_units=100, W=W)

# Case 3: Numpy array
W_init = np.random.normaln
l = lasagne.layers.DenseLayer(l_in, num_units=100, W=W_init)

# What's the parameter structure in layers?
params = lasagne.layers.get_all_param_values(l_hidden) # A list
W = params[0] # array (50,100) The shape of the matrix for W should be (num_inputs, num_units)
b = params[1] # array (200,1)


# Get output 
# Propagating data through layers
# Method 1
y = lasagne.layers.get_output(l_out, deterministic=True)
f = theano.function([l_in.input_var], lasagne.layers.get_output(l_out))

# Method 2
x = T.matrix('x')
y = lasagne.layers.get_output(l_out, x)
f = theano.function([x], y)

# Helpful! 
# http://lasagne.readthedocs.io/en/latest/modules/layers.html


# How to initiation
# How to build weight matrix
# The shape of weight is a 4D tensor: (num_filters, num_input_channels, filter_rows, filter_columns)
'''
W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 4D tensor with shape
        ``(num_filters, num_input_channels, filter_rows, filter_columns)``.
        See :func:`lasagne.utils.create_param` for more information.
'''
# Intuitive
w1 = np.ones([3,3])
w1 = w1.reshape(-1,3,3)
w2 = np.ones([3,3])*2
w2 = w2.reshape(-1,3,3)
W_filter1 = np.concatenate((w1,w2),axis=0)

w1 = np.ones([3,3])
w1 = w1.reshape(-1,3,3)
w2 = np.ones([3,3])*2
w2 = w2.reshape(-1,3,3)
W_filter2 = np.concatenate((w1,w2),axis=0)

w1 = np.ones([3,3])
w1 = w1.reshape(-1,3,3)
w2 = np.ones([3,3])*2
w2 = w2.reshape(-1,3,3)
W_filter3 = np.concatenate((w1,w2),axis=0)

Ws = np.concatenate((W_filter1.reshape(-1,2,3,3),W_filter2.reshape(-1,2,3,3),W_filter3.reshape(-1,2,3,3)),axis=0)

# Wrap-up
# if we want to initial weight of [2,2,3,3]
shape = [2,2,3,3]
[NumChannel,NumFilter,size,size] = shape

Ws = np.array([]).reshape(1,-1)
for i in range (0,NumFilter*NumFilter):
    w = np.ones([3,3]).reshape(1,-1) # np.ones([3,3]) can be replaced by other functions
    Ws = np.concatenate((Ws,w),axis=1)
Ws = Ws.reshape(2,2,3,3)
Ws
