 # training implementation of unet.py
import theano
import lasagne
import random
import numpy as np
import scipy.ndimage.interpolation
import pickle
import imp
from theano import tensor as T
import unet
import elastic_transform_square as elastic

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
    max_rotation = 30 #must be intuitive
    max_shift = 20 #must be relevant to total size (1/10 is good)
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
    shape = train_input_var.shape
    test_shape = test_input_var.shape

    input_var = T.tensor4('input_var')  
    label_var = T.tensor4('label_var')

    [network, loss, test_loss, test_acc, output_det] = unet.network(input_var, label_var, [batch,1,shape[2],shape[3]])

    params = lasagne.layers.get_all_params(network)
    lr = theano.shared(lasagne.utils.floatX(1e-4)) # learning rate
    updates = lasagne.updates.adam(loss,params, learning_rate=lr) # adam most widely used update scheme
    #updates = lasagne.updates.momentum(loss,params, learning_rate=lr,momentum=0.99)

    train_fn = theano.function([input_var, label_var], loss, updates=updates, allow_input_downcast=True) #update weights, #allow_input_downcast for running 32-bit theano on 64-bit machine, might not need
    test_fn = theano.function([input_var, label_var], test_loss, allow_input_downcast=True) #update weights, #allow_input_downcast for running 32-bit theano on 64-bit machine, might not need
    
    best = None
    best_test_err = 10000

    train_input_new = train_input_var
    train_label_new = train_label_var

    # Start training the network
    log_file = 'log.txt'
    f = open(log_file, 'w')

    while (epoch < 501):
        train_err = run_epoch(train_input_new, train_label_new, train_fn, shape)
        test_err = run_epoch(test_input_var, test_label_var, test_fn, test_shape)
        print(test_err,train_err)
        f.write(str(test_err)+'\t'+str(train_err)+'\n')

        if test_err < best_test_err:
            best_test_err = test_err
            best_epoch = epoch
            best = [np.copy(p) for p in (lasagne.layers.get_all_param_values(network))]
            
        if epoch%10 == 0:
            params_file='params_epoch_'+str(epoch)
            with open(params_file, 'wb') as wr:
                pickle.dump(best, wr)
                pass
            wr.close()

        print('%d epoch finished' %(epoch))
        epoch = epoch+1

        train_input_new, train_label_new = image_transform(train_input_var, train_label_var, shape)

    f.close()

        

