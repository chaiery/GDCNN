import theano
import lasagne_dev as nn
import random
import numpy as np
import scipy.ndimage.interpolation
import pickle
import imp
from theano import tensor as T
import unet
import elastic_transform_square as elastic
from collections import OrderedDict
from g_gradient import *


batch = 5 #16 images fed in each time
max_rotation = 30 #must be intuitive
max_shift = 20 #must be relevant to total size (1/10 is good)

def run_params(train_input_var, train_label_var, test_input_var, test_label_var):
    shape = train_input_var.shape
    test_shape = test_input_var.shape

    input_var = train_input_var
    label_var = train_label_var

    [network, loss, test_loss, test_acc, output_det] = unet.network(input_var, label_var, [batch,1,shape[2],shape[3]])


    params = nn.layers.get_all_params(network)
    lr = theano.shared(nn.utils.floatX(1e-4)) # learning rate
    updates = nn.updates.adam(loss,params, learning_rate=lr) # adam most widely used update scheme

    gs = nn.layers.get_all_gs(network)

    gs_updates = g_updates(loss, params, gs)
    #value = [gs_updates[gs[i]].get_value() for i in range (0,len(gs))]

    ## generate updates for params
    updates = OrderedDict()
    updates_old = nn.updates.adam(loss, params, learning_rate=lr)
    for i in range (0,len(gs)):
        print(i)
        gs_new = gs_updates[gs[i]]
        ws = params[i*2]
        [num_filters, num_channels, filter_size, filter_size] = ws.get_value().shape
        W = gabor_weight_update([num_filters, num_channels, filter_size, filter_size], gs_new)
        #updates[ws] = theano.shared(W)
        updates[ws] = W
        updates[params[i*2+1]] = updates_old[params[i*2+1]]

    for key in updates.keys():
    	key.set_value = updates[key]

    '''
    params = lasagne.layers.get_all_params(network)
    lr = theano.shared(lasagne.utils.floatX(1e-4)) # learning rate
    updates = lasagne.updates.sgd(loss,params, learning_rate=lr) # adam most widely used update scheme
    #updates = lasagne.updates.momentum(loss,params, learning_rate=lr,momentum=0.99)
    '''



    print(test_loss.eval(),loss.eval())
