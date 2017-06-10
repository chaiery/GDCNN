 # training implementation of unet.py
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
import timeit

batch = 5 #16 images fed in each time
max_rotation = 30 #must be intuitive
max_shift = 20 #must be relevant to total size (1/10 is good)


def run_epoch(input_var, label_var, fn, shape):
    count = int(shape[0]/batch) # number of total batches
    i = 0
    err = 0

    while (i<count):
        start = i*batch
        tmp_input = input_var[i*batch:(i+1)*batch]
        tmp_label = label_var[i*batch:(i+1)*batch]

        e = fn(tmp_input, tmp_label)
        err += e
        i += 1 # i is batch number

        pass

    err = err/count #normalize error
    
    return err


def image_transform(input_var, label_var, shape):
    max_rotation = 40 #must be intuitive
    max_shift = 30 #must be relevant to total size (1/10 is good)
    tmp_input = []
    tmp_label = []
    for start in range (0, input_var.shape[0]):
        img=input_var[start,0,:,:]
        img=img.reshape(shape[2],shape[3])
        #rotate image
        r = (random.random()-0.5)*2
        r_rotate = r*max_rotation # only rotate training images
        img=scipy.ndimage.interpolation.rotate(img, r_rotate, reshape=False)
        #shift
        r = (random.random()-0.5)*2
        r_shift_x = r*max_shift
        r = (random.random()-0.5)*2
        r_shift_y = r*max_shift
        img = scipy.ndimage.interpolation.shift(img, [r_shift_x, r_shift_y])
        
        label = label_var[start,0,:,:]
        label = label.reshape(shape[2],shape[3])
        label = scipy.ndimage.interpolation.rotate(label, r_rotate, reshape=False)
        label = scipy.ndimage.interpolation.shift(label, [r_shift_x, r_shift_y])

        
        #elastic_transform
        img, label = elastic.elastic_transform(img, label, 10, 2, random_state=None)
       
        img = img.reshape(1,shape[2],shape[3]) #reshape the image to original size afer shift
        label = label.reshape(1,shape[2],shape[3]) #reshape the image to original size afer shift
        
        tmp_input.append(img)
        tmp_label.append(label)
    
    return np.asarray(tmp_input), np.asarray(tmp_label)


def run_params(train_input_var, train_label_var, test_input_var, test_label_var):
    epoch = 1
    rand = 1
    shape = train_input_var.shape
    test_shape = test_input_var.shape

    input_var = T.tensor4('input_var')  
    label_var = T.tensor4('label_var')

    [network, loss, test_loss, test_acc, output_det] = unet.network(input_var, label_var, [batch,1,shape[2],shape[3]])


    params = nn.layers.get_all_params(network)
    lr = theano.shared(nn.utils.floatX(1e-4)) # learning rate
    

    lr_g = theano.shared(nn.utils.floatX(1e-10))
    #if epoch%5 == 0
        #lr_g = theano.shared(nn.utils.floatX(lr_g/epoch))
    updates = nn.updates.adam(loss,params[0:len(params)], learning_rate=lr) # adam most widely used update scheme
    '''
    gs = nn.layers.get_all_gs(network)

    gs_updates = g_updates(loss, params, gs, rand, lr_g)

    ## generate updates for params
    updates = OrderedDict()
    updates_old = nn.updates.adam(loss, params, learning_rate=lr)
    for i in range (0,len(gs)):
        print(i)
        gs_new = gs_updates[gs[i]]
        #gs_new = gs[i]
        ws = params[i*2]
        [num_filters, num_channels, filter_size, filter_size] = ws.get_value().shape
        W = gabor_weight_update([num_filters, num_channels, filter_size, filter_size], gs_new)
        #updates[ws] = theano.shared(W)
        updates[ws] = ws
        updates_old[ws] = ws
        updates[params[i*2+1]] = updates_old[params[i*2+1]]
        updates[gs[i]] = gs_new

    for j in range (2*len(gs),len(params)):
        updates[params[j]] = updates_old[params[j]]

    for j in range(0,len(gs)):
        updates_old[gs[j]] = gs_updates[gs[i]]
    '''
    start = timeit.default_timer()
    train_fn = theano.function([input_var, label_var], loss, updates=updates, allow_input_downcast=True) #update weights, #allow_input_downcast for running 32-bit theano on 64-bit machine, might not need
    stop = timeit.default_timer()
    print(stop-start)
    
    start = timeit.default_timer()
    test_fn = theano.function([input_var, label_var], test_loss, allow_input_downcast=True) #update weights, #allow_input_downcast for running 32-bit theano on 64-bit machine, might not need
    stop = timeit.default_timer()
    print(stop-start)

    best = None
    best_test_err = 10000

    i = 0

    train_input_new = train_input_var
    train_label_new = train_label_var

    # Start training the network
    log_file = 'log.txt'
    f = open(log_file, 'w')

    done_looping = False
    while (epoch < 201) and (not done_looping):
        train_err = run_epoch(train_input_new, train_label_new, train_fn, shape)
        test_err = run_epoch(test_input_var, test_label_var, test_fn, test_shape)
        print(test_err,train_err)
        #gs = nn.layers.get_all_gs(network)
        f.write(str(test_err)+'\t'+str(train_err)+'\n')
        #ps = nn.layers.get_all_param_values(network)
        #print(ps[0][0,0,:,:])
        if test_err < best_test_err:
            best_test_err = test_err
            best_epoch = epoch
            best = [np.copy(p) for p in (nn.layers.get_all_param_values(network))]
        
        '''
        if epoch%50 == 0:
            params_file='params_epoch_'+str(epoch)
            with open(params_file, 'wb') as wr:
                pickle.dump(best, wr)
                pass
            wr.close()
	    '''

        print('%d epoch finished' %(epoch))

        if (epoch-best_epoch)>=10:
            Dir = '/home/Documents/params_saving/'
            params_file=Dir+'params_epoch_'+str(best_epoch)
            with open(params_file, 'wb') as wr:
                pickle.dump(best, wr)
                pass
            wr.close()
            print("Best Validation Error is "+str(best_test_err))
            done_looping = True
        epoch = epoch+1

        train_input_new, train_label_new = image_transform(train_input_var, train_label_var, shape)
    
    f.close()

        

