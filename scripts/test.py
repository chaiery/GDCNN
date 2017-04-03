from PIL import Image
import pickle
from input_process import read_images
import lasagne as nn
import theano
from theano import tensor as T
import unet

size = (116, 116) 
pred_path = '../pred/'

pkl_file = open('../fnames_collection', 'rb')
fnames = pickle.load(pkl_file)
for i in range (0,len(fnames)):
	fnames[i] = str(fnames[i])
	fnames[i] = 'image' + fnames[i] + '.png'

test_index = fnames[400:410]
label_path = '../newPNGlabeled/'
ori_path = '../JpegOriginalImg/'
test_input, test_label = read_images(label_path, ori_path, test_index, size)


pkl_file = open('params_epoch_400', 'rb')
params = pickle.load(pkl_file)

input_var = T.tensor4('input_var')   # the data is presented as rasterized images
label_var = T.tensor4('label_var')
[network, loss, test_loss, test_acc, output_det] = unet.network(input_var, label_var, [10,1,size[0],size[1]])

nn.layers.set_all_param_values(network, params)

test_fn=theano.function([input_var, label_var], [test_acc, output_det], allow_input_downcast=True)
test_acc, output_det = test_fn(test_input,test_label)

error = -2*T.sum(output_det*test_label)/T.sum(output_det+test_label+0.0001)
print(error.eval())

for i in range (0,10):
	image_label = test_label[i,0,:,:]
	image_label = Image.fromarray(image_label*255)	

	image_pred = output_det[i,0,:,:]
	image_pred = Image.fromarray(image_pred*255)

	#image_new.show()
	filename_pred = pred_path + test_index[i]

	# Convert the image to RGB:
	if image_pred.mode != 'RGB':
	    image_pred = image_pred.convert('RGB')
	filename_label = pred_path + '/label' + test_index[i]

	if image_label.mode != 'RGB':
	    image_label = image_label.convert('RGB')

	image_pred.save(filename_pred)
	image_label.save(filename_label)



'''
image_1 = output_det[0,0,:,:]


image_new = Image.fromarray(image_1)
image_new = Image.fromarray(image_1*255)
image_new.show()
print(test_acc)

for i in range (0,4):
	image_1 = output_det[i,0,:,:]

	image_1[image_1>0.5] = 1
	image_1[image_1<0.5] = 0

	image_new = Image.fromarray(image_1)
	image_new = Image.fromarray(image_1*255)
	image_new.show()

	
inputlabel = test_label[i,0,:,:]
result = [imtest.reshape(1,-1)==inputlabel.reshape(1,-1)]
'''

'''
image_1 = images[0,0,:,:]
image_new = Image.fromarray(image_1*256)
image_new.show()
'''

'''
test_image_fn = theano.function([input_var], output_det, allow_input_downcast=True)
test_input_var = train_input_var[0:1,:,:,:]
output_det_final = test_image_fn(test_input_var)
images_file='images'

with open(images_file, 'wb') as wr:
    pickle.dump(output_det_final, wr)
    pass
wr.close()
'''
