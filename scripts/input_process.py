# Construct training dataset and test dataset
# input_var = your input image
# 4D: 1 - index of image (0,1,2,...,16)
#     2 - 1
#     3 - actual image (Y)
#     4 - actual image (X)
# label_var = also 4D
#     1 - image IDs (0,1,2,3, ...,16 )
#     2 - 1
#     3 - label image (Y)
#     4 - label image (X)
# shape = gives shape of inputs (use this as a sanity check for expected image size)
#     [16, 1, 256, 256,]

from PIL import Image
import os
import numpy
from scipy import ndimage, misc
from theano import shared

def read_images(label_path, ori_path, data, size):

	input_label = numpy.zeros((len(data),1,size[0],size[1]), dtype='float32')
	input_var = numpy.zeros((len(data),1,size[0],size[1]), dtype ='float32')

	for i in range (0,len(data)):
		fname_label = data[i]
		fname_ori = data[i][0:-4] + '.jpg'

		img = Image.open(label_path+fname_label)
		img = numpy.asarray(img, dtype='float32') / 153.
		img = img[0:116,6:122]
		input_label[i,0,:,:] = img

		img = Image.open(ori_path+fname_ori)
		img = numpy.asarray(img, dtype='float32') / 255.
		img = img[0:116,6:122]
		input_var[i,0,:,:] = img
	shape = input_var.shape

	return input_var, input_label, shape






	

