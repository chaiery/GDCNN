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
    z2 = 1j*2*math.pi*f*xt+psi
    value = f**2/(math.pi*gamma)*math.exp(z1)*cmath.exp(z2)
    return value


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


def g_psi(x,y,params):
    f,gamma,sigma,theta,psi = params
    value = gabor_filter(x,y,params)  
    return value


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



def g_updates(loss, params, gs):
	gs_gradients = []
	gradients = nn.updates.get_or_compute_grads(loss, params)

	for w_index in range (0, int(len(params)/2)):
		# First Loop
		# print(w_index)
		ws = params[w_index*2]
		g_params = gs[w_index]	
		ws_gradients = gradients[w_index*2]

		[num_filters, num_channels, filter_size, filter_size] = np.ndarray.tolist(ws.shape.eval())

		# Second and third Loop
		g_gradients = np.array([], dtype=np.float32).reshape(1,-1)

		for i in range (0, num_filters):
			for j in range (0, num_channels):
				g = g_params[i,j,:]
				w = ws[i,j,:,:]

				w_gradient = ws_gradients[i,j,:]
				
				#addition = gabor_filter_update(filter_size, g.eval(), g_psi)	# need adjustment

				'''
				a = (w_gradient*addition).sum()
				psi_gradient = a/9
				g_gradient = np.array([0,0,0,0,psi_gradient.eval()]).reshape(1,-1)		# need adjustment				
				g_gradients = np.concatenate((g_gradients,g_gradient),axis=1)
				'''
				#a = sum(sum(w_gradient*addition))
				#a = (w_gradient*addition).sum()
				#psi_gradient = a/9
				g_gradient = np.array([0,0,0,0,0]).reshape(1,-1)
				#g_gradient = np.array([0,0,0,0,psi_gradient]).reshape(1,-1)		# need adjustment				
				g_gradients = np.concatenate((g_gradients,g_gradient),axis=1)

		g_gradients = g_gradients.reshape([num_filters,num_channels,-1])
		
		#g_gradients = theano.shared(nn.utils.floatX(g_gradients))
		g_gradients = theano.shared(g_gradients)

		gs_gradients.append(g_gradients)

	gs_updates = nn.updates.adam_dev(gs_gradients, gs)

	return gs_updates
