from PIL import Image
import pickle
from input_process import read_images
import lasagne_dev as nn
import theano
from theano import tensor as T
import unet

size = (116, 116) 
pred_path = '../Pred/'
batch = 1000

pkl_file = open('../fnames_collection_2', 'r')
fnames = pkl_file.readline()
fnames = fnames.split(' ')
for i in range (0,len(fnames)):
	fnames[i] = str(fnames[i])
	fnames[i] = 'image' + fnames[i] + '.png'


test_index = fnames[4000:5000]

label_path = '../NewPNGlabeled/'
ori_path = '../JpegOriginalImg/'

test_input, test_label= read_images(label_path, ori_path, test_index, size)



pkl_file = open('/home/spc/Documents/params_saving/params_epoch_27_gabor7_ast', 'rb')
params = pickle.load(pkl_file)

input_var = T.tensor4('input_var')   # the data is presented as rasterized images
label_var = T.tensor4('label_var')
[network, loss, test_loss, test_acc, output_det] = unet.network(input_var, label_var, [batch,1,size[0],size[1]])

nn.layers.set_all_param_values(network, params)

test_fn=theano.function([input_var, label_var], [test_loss, output_det], allow_input_downcast=True)


count = int(test_input.shape[0])/batch
i = 0
error = 0
loop = 0
while loop<count:

	test_loss, output_det = test_fn(test_input[loop*batch:(loop+1)*batch],test_label[loop*batch:(loop+1)*batch])

	error += test_loss

	for i in range (0,batch):
		image_pred = output_det[i,0,:,:]
		image_pred = Image.fromarray(image_pred*255)

		#image_new.show()
		filename_pred = pred_path + test_index[i+loop*batch]

		# Convert the image to RGB:
		if image_pred.mode != 'RGB':
		    image_pred = image_pred.convert('RGB')

		image_pred.save(filename_pred)

	loop += 1

error = error/count
print(error)

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
