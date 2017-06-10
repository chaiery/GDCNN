from theano import tensor as T
import lasagne_dev as nn
import numpy as np
import math
from initiation import *
# lasagne should be the latest (under development verison)
# sudo pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
# sudo pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
# sudo pip install lasagne Pillow

# input_var = your input image
# 4D: 1 - index of image (0,1,2,...,batch_size)
#     2 - channel (RGB: 3, BW: 1)
#     3 - actual image (Y)
#     4 - actual image (X)
# label_var = also 4D
#     1 - image IDs (0,1,2,3, ...,batch_size )
#     2 - 1
#     3 - label image (Y)
#     4 - label image (X)
# shape = gives shape of inputs (use this as a sanity check for expected image size)
#     [16, 1, 272, 272,]

# size = (116, 116)

def sorenson_dice(pred, target):
    # 2 is just a scaling factor
    # add a number at end (20) to avoid dividing by 0
    return -2*T.sum(pred*target)/T.sum(pred+target+0.0001)

def gabor_filter(x,y,w,theta,sigma):
    xt = x*np.cos(theta) + y*np.sin(theta)
    yt = -x*np.sin(theta) + y*np.cos(theta)
    
    z1 = -(xt**2 + yt**2)/(2*sigma**2)
    z2 = 1j*w*xt + w**2*sigma**2/2
    value = (1/(2*math.pi*sigma**2)*np.exp(z1)*np.exp(z2)).real
    value = value.astype(np.float32)
    return value 


def rescale(gfilter,mag):
    mi = np.min(gfilter)
    ma = np.max(gfilter)
    factor = 0.3/max([ma,-mi])
    return gfilter*factor


def gabor32_filter_initiation(shape):
    [num_filters,num_channels,size,size] = shape
    number = num_filters*num_channels
    Ws = []
    gfilter = []
    
    gs = np.array([],dtype=np.float32).reshape(1,-1)
    n = 8;
    m = int(number/n);
    
    bond = math.floor(size/2)
    x_range = np.linspace(-bond, bond, size)
    y_range = np.linspace(-bond, bond, size)
    xt,yt = np.meshgrid(x_range,y_range)
    
    for i in range (1,n+1):
        for j in range (1,m+1):
            w = (math.pi/2)*(2**0.5)**(-j+1)
            theta = (i-1)*math.pi/8
            sigma = math.pi/w

            a = gabor_filter(xt,yt,w,theta,sigma).reshape(1,size,size)
            #a = rescale(a,mag=0.3)
            if len(gfilter)==0:
                gfilter = a
            else:
                gfilter = np.concatenate((gfilter,a),axis=0)

    Ws = gfilter.reshape(shape)
            
    return Ws

def network(input_var, label_var, shape):
    layer = nn.layers.InputLayer(shape,input_var) #input layer (image size 116)
    
    #convolution layers (32 filters)
    #nonlinearality = nn.nonlinearities.rectify <-- ReLu
    layer = nn.layers.Conv2DLayer(layer, num_filters = 32,filter_size = 7, W = gabor32_filter_initiation([32,1,7,7]),
                                  nonlinearity = nn.nonlinearities.rectify, pad='same') #112
    
    #layer = nn.layers.Conv2DLayer(layer, num_filters = 32,filter_size = 7,
    #                             nonlinearity = nn.nonlinearities.rectify, pad='same') #112
    # max pool layer (stride = 2)
    layer = nn.layers.MaxPool2DLayer(layer, pool_size = 2) #56 (half of previous layer)
    
    #convolution layers (64 filters)
    layer = nn.layers.Conv2DLayer(layer, num_filters = 64,filter_size = 7,
                                  nonlinearity = nn.nonlinearities.rectify, pad='same') #52

    # max pool layer
    layer = nn.layers.MaxPool2DLayer(layer, pool_size = 2) #26 (half of previous layer)
    
    #convolution layers (256 filters)
    layer = nn.layers.Conv2DLayer(layer, num_filters = 128,filter_size = 7,
                                  nonlinearity = nn.nonlinearities.rectify, pad='same') #22
                 
    #upscale layer
    layer = nn.layers.Upscale2DLayer(layer, scale_factor = 2) #52


    #convolutional layers with 'full' pad
    layer = nn.layers.Conv2DLayer(layer, num_filters = 64,filter_size = 7,
                                  nonlinearity = nn.nonlinearities.rectify, pad = 'same') #26

    #upscale layer
    layer = nn.layers.Upscale2DLayer(layer, scale_factor = 2) #112    

    #convolutional layers with 'full' pad
    layer = nn.layers.Conv2DLayer(layer, num_filters = 32,filter_size = 7,
                                  nonlinearity = nn.nonlinearities.rectify, pad = 'same') #56


    #convolutional layers with 'full' pad
    layer = nn.layers.Conv2DLayer(layer, num_filters = 1,filter_size = 7,
                                  nonlinearity = nn.nonlinearities.sigmoid, pad = 'same') #116 


    # this gives the labels for the image that was fed into the nn
    # get_output can be used for any layer, not just final
    output = nn.layers.get_output(layer) # parameters will be updated - use for train
    output_det = nn.layers.get_output(layer, deterministic=True) # no backpropagation, params are fixed, use for testing
    loss = sorenson_dice(output, label_var) #for training loss
    test_loss = sorenson_dice(output_det, label_var) #for test loss
    test_acc = nn.objectives.binary_accuracy(output_det, label_var)

    return layer, loss, test_loss, test_acc, output_det
    
    
