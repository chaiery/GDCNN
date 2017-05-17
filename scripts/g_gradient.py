import lasagne_dev as nn
import numpy as np
import math
import cmath
import theano


def gabor_filter(x,y,params):
    f,gamma,sigma,theta,psi = params
    xt = x*math.cos(theta) + y*math.sin(theta)
    yt = -x*math.sin(theta) + y*math.cos(theta)
    z1 = -(xt**2 + (gamma*yt)**2)/(2*sigma**2)
    z2 = 2*math.pi*f*xt+psi
    value = f**2/(math.pi*gamma)*math.exp(z1)*math.cos(z2)
    return value

def gabor_filter_dev(x,y,params):
    f,gamma,sigma,theta,psi = params
    z2 = psi
    value = f**2/(math.pi*gamma)*math.cos(z2)
    return value

def gabor_filter_tensor(x,y,params):
    f,gamma,sigma,theta,psi = params
    xt = x*theano.tensor.cos(theta) + y*theano.tensor.sin(theta)
    yt = -x*theano.tensor.sin(theta) + y*theano.tensor.cos(theta)
    z1 = -(xt**2 + (gamma*yt)**2)/(2*sigma**2)
    z2 = 2*math.pi*f*xt+psi
    #z2 = 1j*2*math.pi*f*xt+psi
    value = f**2/(math.pi*gamma)*theano.tensor.exp(z1)*theano.tensor.cos(z2)
    return value

'''
def g_theta(x,y,params):
    f,gamma,sigma,theta,psi = params
    value = gabor_filter(x,y,params)
    return ((yt*xt/sigma**2)*(gamma**2-1)+1j*2*math.pi*f*yt)*value


def g_f(x,y,params):
    f,gamma,sigma,theta,psi = params
    value = gabor_filter(x,y,params)  
    return (2/f + 1j*2*math.pi*xt)*value


def g_gamma(x,y,params):
    f,gamma,sigma,theta,psi = params
    value = gabor_filter(x,y,params)  
    return -(yt**2*gamma/sigma**2+1/gamma)*value


def g_theta(x,y,params):
    f,gamma,sigma,theta,psi = params
    value = gabor_filter(x,y,params)  
    return ((xt**2 + gamma**2*yt**2)/sigma**3)*value
'''

def g_psi(x,y,params):
    f,gamma,sigma,theta,psi = params
    value = gabor_filter_dev(x,y,params)  
    return value


def g_updates(loss, params, gs):
    # Calculate gradients: 0.05
    gs_gradients = []
    gradients = nn.updates.get_or_compute_grads(loss, params)
    
    for w_index in range (0, int(len(params)/2)):
        # First Loop
        # print(w_index)
        
        ws = params[w_index*2]
        g_params = gs[w_index].get_value()
        ws_gradients = gradients[w_index*2]

        [num_filters, num_channels, filter_size, filter_size] = ws.get_value().shape
        
        position = filter_size/2
        ws_grad = gradients[0][:,:,position:position+1,position]

        # Second and third Loop
        additions = []
        for i in range (0, num_filters):
            for j in range (0, num_channels):
                g = g_params[i,j,:]
                additions.append(g_psi(0,0,g))
                
        additions =  np.array(additions).reshape(num_filters,num_channels,-1)
        g_gradients = theano.tensor.concatenate([ws_grad*0, ws_grad*0, ws_grad*0, ws_grad*0, ws_grad*0], axis=2)*additions
        gs_gradients.append(g_gradients)
    gs_updates = nn.updates.adam_dev(gs_gradients, gs)
    return gs_updates


def gabor_weight_update(shape, gs):
    [num_filters,num_channels,size,size] = shape
    #Ws = np.array([], dtype=np.float32).reshape(1,-1)

    gfilter = []

    for filter_index in range (0,num_filters):
        for channel_index in range (0,num_channels):
            params = gs[filter_index, channel_index]
            g_params = [params[0],params[1],params[2],params[3],params[4]]

        
            bond = math.floor(size/2)
            x_range = np.linspace(-bond, bond, size)
            y_range = np.linspace(-bond, bond, size)

            [x_range,y_range] = list(map(lambda x:x.reshape(1,-1),np.meshgrid(x_range,y_range)))
            
            for (x,y) in zip(np.ndarray.tolist(x_range)[0], np.ndarray.tolist(y_range)[0]):
                value = gabor_filter_tensor(x,y,g_params)
                gfilter.append(value)
            
    Ws = theano.tensor.stack(gfilter)
    Ws = W.reshape(shape)
            #W = np.array(gfilter, dtype=np.float32)
            #W = np.array(gfilter)
            #W = W.reshape(1,-1)
            #Ws = np.concatenate((Ws,W),axis=1)
    return Ws



'''
def g_updates_dev(loss, params, gs):
    # Calculate gradients: 0.05
    gs_gradients = []
    gradients = nn.updates.get_or_compute_grads(loss, params)
    
    for w_index in range (0, int(len(params)/2)):
        # First Loop
        # print(w_index)
        
        ws = params[w_index*2]
        g_params = gs[w_index].get_value()
        ws_gradients = gradients[w_index*2]

        [num_filters, num_channels, filter_size, filter_size] = ws.get_value().shape

        g_gradients = np.array([], dtype=np.float32).reshape(1,-1)
    

        # Second and third Loop

        for i in range (0, num_filters):
            for j in range (0, num_channels):
                
                g = g_params[i,j,:]
                w = ws[i,j,:,:]

                w_gradient = ws_gradients[i,j,:]
                
                update = g_psi(0,0,g)
                #addition = gabor_filter_update_dev(filter_size, g, g_psi)	# need adjustment
                
                
                #a = (w_gradient*addition).sum()
                #psi_gradient = a/9
                #g_gradient = np.array([0,0,0,0,psi_gradient.eval()]).reshape(1,-1)		# need adjustment				
                #g_gradients = np.concatenate((g_gradients,g_gradient),axis=1)
                
                #a = sum(sum(w_gradient*addition))
                #a = (w_gradient*addition).sum()
                #psi_gradient = a/9
                g_gradient = np.array([0,0,0,0,update]).reshape(1,-1)
                #g_gradient = np.array([0,0,0,0,psi_gradient]).reshape(1,-1)
                g_gradients = np.concatenate((g_gradients,g_gradient),axis=1)
                

        
        g_gradients = g_gradients.reshape([num_filters,num_channels,-1])
        
        #g_gradients = theano.shared(nn.utils.floatX(g_gradients))
        g_gradients = theano.shared(g_gradients)

        gs_gradients.append(g_gradients)
        
    
    #gs_updates = nn.updates.adam_dev(gs_gradients, gs)
    gs_updates = nn.updates.adam_dev(gradients, params)
    return 0

def gabor_filter_update(size, params, f):
    bond = math.floor(size/2)
    x_range = np.linspace(-bond, bond, size)
    y_range = np.linspace(-bond, bond, size)

    [x_range,y_range] = list(map(lambda x:x.reshape(1,-1),np.meshgrid(x_range,y_range)))
    gfilter = []
    updates = []
    for (x,y) in zip(np.ndarray.tolist(x_range)[0], np.ndarray.tolist(y_range)[0]):        
        update = f(x,y,params)
        updates.append(update.real)
    updates = np.array(updates, dtype=np.float32).reshape(size,-1)
    return updates
'''
