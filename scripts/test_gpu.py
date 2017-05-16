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


input_var = T.tensor4('input_var')  
label_var = T.tensor4('label_var')

[network, loss, test_loss, test_acc, output_det] = unet.network(input_var, label_var, [5,1,116,116])


params = nn.layers.get_all_params(network)
lr = theano.shared(nn.utils.floatX(1e-4)) # learning rate
updates = nn.updates.adam(loss,params, learning_rate=lr) # adam most widely used update scheme
#updates = nn.updates.momentum(loss,params, learning_rate=lr,momentum=0.99)
gs = nn.layers.get_all_gs(network)

