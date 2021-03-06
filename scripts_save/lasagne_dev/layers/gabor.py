import theano.tensor as T
import random
import math
import cmath
import numpy as np

from .. import init
from .. import nonlinearities
from ..utils import as_tuple
from ..theano_extensions import conv

from .conv import Conv2DLayer
from .. import utils


__all__ = [
    "gabor_Conv2DLayer"
]


def gabor_filter(x,y,params):
    f,gamma,sigma,theta,psi = params
    xt = x*np.cos(theta) + y*np.sin(theta)
    yt = -x*np.sin(theta) + y*np.cos(theta)
    z1 = -(xt**2 + (gamma*yt)**2)/(2*sigma**2)
    z2 = 2*math.pi*f*xt+psi
    value = f**2/(math.pi*gamma)*np.exp(z1)*np.cos(z2)
    value = value.astype(np.float32)
    return value

def random_gabor(shape):
    [NumFilter, NumChannel] = shape
    gs = np.array([],dtype=np.float32).reshape(1,-1)

    for i in range (0,NumChannel*NumFilter):
        gamma = random.uniform(0.0001,10)
        sigma = random.uniform(0.0001,5)
        theta = random.uniform(0,2*math.pi)    
        psi = random.uniform(0,0.915)
        f = random.uniform(-7.85398*gamma,7.85398*gamma)

        params = [f, gamma, sigma, theta, psi]
        
        g = np.array(params, dtype=np.float32).reshape(1,-1)
        gs = np.concatenate((gs,g),axis=1)
    gs = gs.reshape([NumFilter, NumChannel, 5])
    return gs  


def gabor_filter_initiation(shape, gs):
    [num_filters,num_channels,size,size] = shape
    Ws = []
    gfilter = []

    for filter_index in range (0,num_filters):
        for channel_index in range (0,num_channels):
            
            params = gs[filter_index, channel_index]
            params = np.ndarray.tolist(params)
            
            bond = math.floor(size/2)
            x_range = np.linspace(-bond, bond, size)
            y_range = np.linspace(-bond, bond, size)

            xt,yt = np.meshgrid(x_range,y_range)
            a = gabor_filter(xt,yt,params).reshape(1,size,size)
            if len(gfilter)==0:
                gfilter = a
            else:
                gfilter = np.concatenate((gfilter,a),axis=0)

        gfilter = gfilter.reshape(1,-1,size,size)

        if len(Ws)==0:
            Ws = gfilter 
        else:
            Ws = np.concatenate((Ws,gfilter),axis=0)
        gfilter = []
            
    return Ws


class gabor_Conv2DLayer(Conv2DLayer):
    """
    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.

    g : Theano shared variable or expression
        Variable or expression representing the paramters of gabor filters.
    """


    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 b=init.Constant(0.), g_f=random_gabor,
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 convolution=T.nnet.conv2d, **kwargs):

        num_channels = incoming.output_shape[1]

        gs = g_f([num_filters, num_channels])

        W = gabor_filter_initiation([num_filters, num_channels, filter_size, filter_size], gs)

        super(gabor_Conv2DLayer, self).__init__(incoming, num_filters, filter_size, stride,
                                                 pad, untie_biases, W, b, nonlinearity, flip_filters,
                                                 convolution, **kwargs)

        g_shape = [num_filters, num_channels, 5]
        self.g = utils.create_param(gs, g_shape, 'g')



'''
def gabor_filter_initiation(shape, gs):
    [NumFilter,NumChannel,size,size] = shape
    Ws = np.array([], dtype=np.float32).reshape(1,-1)
    for filter_index in range (0,NumFilter):
        for channel_index in range (0,NumChannel):
            params = gs[filter_index, channel_index]
            params = np.ndarray.tolist(params)

            bond = math.floor(size/2)
            x_range = np.linspace(-bond, bond, size)
            y_range = np.linspace(-bond, bond, size)

            [x_range,y_range] = list(map(lambda x:x.reshape(1,-1),np.meshgrid(x_range,y_range)))
            gfilter = []
            for (x,y) in zip(np.ndarray.tolist(x_range)[0], np.ndarray.tolist(y_range)[0]):
                value = gabor_filter(x,y,params)
                gfilter.append(value)
            
            W = np.array(gfilter, dtype=np.float32)
            W = W.reshape(1,-1)
            Ws = np.concatenate((Ws,W),axis=1)
    Ws = Ws.reshape(shape)
    return Ws

def gabor_filter(x,y,params):
    f,gamma,sigma,theta,psi = params
    xt = x*math.cos(theta) + y*math.sin(theta)
    yt = -x*math.sin(theta) + y*math.cos(theta)
    z1 = -(xt**2 + (gamma*yt)**2)/(2*sigma**2)
    z2 = 2*math.pi*f*xt+psi
    value = f**2/(math.pi*gamma)*math.exp(z1)*math.cos(z2)
    return value
'''


