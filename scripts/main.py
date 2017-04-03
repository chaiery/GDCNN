from input_process import read_images
from train import run_params
import pickle

size = (116, 116) # resize images

pkl_file = open('../fnames_collection_2', 'r')
fnames = pkl_file.readline()
fnames = fnames.split(' ')
for i in range (0,len(fnames)):
	fnames[i] = str(fnames[i])
	fnames[i] = 'image' + fnames[i] + '.png'

train_index = fnames[1:1000]
test_index = fnames[1301:2000]

label_path = '../NewPNGlabeled/'
ori_path = '../JpegOriginalImg/'
train_input, train_label, shape = read_images(label_path, ori_path, train_index, size)
test_input, test_label, shape = read_images(label_path, ori_path, test_index, size)

run_params(train_input, train_label, test_input, test_label, shape)

# Test images
