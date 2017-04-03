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


# Rough idea using GPU is 
# For each epoch
# Using rotation, elastic transform, shif
# then store in shared
# Go to the function
# A good way to calculate time! Uisng date of epoch*0 files!



batch = 10 #16 images fed in each time
max_rotation = 30 #must be intuitive
max_shift = 20 #must be relevant to total size (1/10 is good)

def run_epoch(fn, shape):
    count = int(shape[0]/batch) # number of total batches
    i = 0
    err = 0
    while (i<count):
        e = fn(i)
        err += e
        i += 1 # i is batch number

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

        #elastic_transform
        img = elastic.elastic_transform(img, 10, 2, random_state=None)

        img = img.reshape(1,shape[2],shape[3]) #reshape the image to original size afer shift

        tmp_input.append(img)
        
        label = label_var[start,0,:,:]
        label = label.reshape(shape[2],shape[3])
        label = scipy.ndimage.interpolation.rotate(label, r_rotate, reshape=False)
        label = scipy.ndimage.interpolation.shift(label, [r_shift_x, r_shift_y])

        label = elastic.elastic_transform(label, 10, 2, random_state=None)
        label = label.reshape(1,shape[2],shape[3]) #reshape the image to original size afer shift

        tmp_label.append(label)

    return np.asarray(tmp_input), np.asarray(tmp_label)
        

def run_params(train_input_var, train_label_var, test_input_var, test_label_var):
    epoch = 1
    index = 0
    test_input_var = theano.shared(test_input_var)
    test_label_var = theano.shared(test_label_var)

    train_input_var_tf = theano.shared(train_input_var)
    train_label_var_tf = theano.shared(train_label_var)

    input_var = T.tensor4('input_var')
    label_var = T.tensor4('label_var')


    shape_train = train_input_var.shape
    shape_test = test_input_var.shape

    [network, loss, test_loss, test_acc, output_det] = unet.network(input_var, label_var, [batch,1,shape_train[2],shape_train[3]])

    params = lasagne.layers.get_all_params(network)
    lr = theano.shared(lasagne.utils.floatX(1e-4)) # learning rate
    updates = lasagne.updates.adam(loss,params, learning_rate=lr) # adam most widely used update scheme
    #updates = lasagne.updates.momentum(loss,params, learning_rate=lr,momentum=0.99)

    train_fn = theano.function(inputs=[index], outputs=loss, updates=updates, allow_input_downcast=True, 
                                givens={ (input_var: train_input_var_tf[index*batch:(index+1)*batch]),
                                         (label_var: train_label_var_tf[index*batch:(index+1)*batch])}) #update weights, #allow_input_downcast for running 32-bit theano on 64-bit machine, might not need
    
    test_fn = theano.function(inputs=[index], outputs=test_loss, allow_input_downcast=True, 
                                givens={ (input_var: test_input_var[index*batch:(index+1)*batch],
                                         label_var: test_label_var[index*batch:(index+1)*batch]}) #update weights, #allow_input_downcast for running 32-bit theano on 64-bit machine, might not need

    best = None
    best_test_err = 10000
    
    # Start training the network
    log_file = 'log.txt'
    f = open(log_file, 'w')

    while (epoch < 501):
        train_err = run_epoch(train_fn, shape_train)
        test_err = run_epoch(train_fn, shape_test)
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
        train_input_new, train_label_new = image_transform(train_input_var, train_label_var, shape_train)
        train_input_var_tf.set_value(train_input_new)
        train_label_var_tf.set_value(train_label_new)

    f.close()

        

