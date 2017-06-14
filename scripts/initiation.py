import numpy as np
import math
import lasagne_dev as nn

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


def gabor_random_filter_initiation(shape):
    [num_filters,num_channels,size,size] = shape

    Ws = []
    gfilter = []
    
    gs = np.array([],dtype=np.float32).reshape(1,-1)
    n = 8;
    m = 4;
    
    bond = math.floor(size/2)
    x_range = np.linspace(-bond, bond, size)
    y_range = np.linspace(-bond, bond, size)
    xt,yt = np.meshgrid(x_range,y_range)
    
    for filter_index in range (0,num_filters):
        sigma = (3-0.1)/num_filters*filter_index + 0.1
        gfilter = []
        for i in range (1,n+1):
            for j in range (1,m+1):
                w = (math.pi/2)*(2**0.5)**(-j+1)
                theta = (i-1)*math.pi/8

                a = gabor_filter(xt,yt,w,theta,sigma).reshape(1,size,size)

                a = rescale(a,mag=0.3)
                if len(gfilter)==0:
                    gfilter = a
                else:
                    gfilter = np.concatenate((gfilter,a),axis=0)

        gfilter = gfilter.reshape(1,-1,size,size)            
        if len(Ws)==0:
            Ws = gfilter 
        else:
            Ws = np.concatenate((Ws,gfilter),axis=0)

    Ws = Ws.reshape(shape)
            
    return Ws


def gabor_32_64_filter_initiation(shape):
    [num_filters,num_channels,size,size] = shape

    Ws = []
    gfilter = []
    
    gs = np.array([],dtype=np.float32).reshape(1,-1)
    n = 8;
    m = 8;
    
    bond = math.floor(size/2)
    x_range = np.linspace(-bond, bond, size)
    y_range = np.linspace(-bond, bond, size)
    xt,yt = np.meshgrid(x_range,y_range)
    
    for filter_index in range (0,num_filters):
        sigma = (3-0.5)/num_filters*filter_index + 0.5
        gfilter = []
        for i in range (1,n+1):
            for j in range (1,m+1):
                w = (math.pi/2)*(2**0.5)**(-j+1)
                theta = (i-1)*math.pi/8

                a = gabor_filter(xt,yt,w,theta,sigma).reshape(1,size,size)

                a = rescale(a,mag=0.3)
                if len(gfilter)==0:
                    gfilter = a
                else:
                    gfilter = np.concatenate((gfilter,a),axis=0)

        gfilter = gfilter.reshape(1,-1,size,size)            
        if len(Ws)==0:
            Ws = gfilter 
        else:
            Ws = np.concatenate((Ws,gfilter),axis=0)

    Ws = Ws.reshape(shape)
            
    return Ws


def gabor_filter_initiation(shape):
    [num_filters,num_channels,size,size] = shape
    number = num_filters*num_channels
    Ws = []
    gfilter = []
    
    gs = np.array([],dtype=np.float32).reshape(1,-1)
    n = 16;
    m = 16;
    
    bond = math.floor(size/2)
    x_range = np.linspace(-bond, bond, size)
    y_range = np.linspace(-bond, bond, size)
    xt,yt = np.meshgrid(x_range,y_range)
    
    for i in range (1,n+1):
        for j in range (1,m+1):
            for z in [1,0.5]:
                w = (math.pi/2)*(2**0.5)**(-j*0.5+1)
                theta = (i-1)*math.pi/16
                sigma = math.pi/w*z

                a = gabor_filter(xt,yt,w,theta,sigma).reshape(1,size,size)
                a = rescale(a,mag=0.3)
                if len(gfilter)==0:
                    gfilter = a
                else:
                    gfilter = np.concatenate((gfilter,a),axis=0)
    
   
    W = nn.init.GlorotUniform()
    a = W([num_filters*num_channels-n*m*2,size,size])
    a = rescale(a,mag=0.3)
    a = a.reshape(-1,size,size)
    gfilter = np.concatenate([gfilter,a],axis=0)

    Ws = gfilter.reshape(shape)
            
    return Ws