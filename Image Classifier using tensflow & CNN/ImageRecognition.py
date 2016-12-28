##################
# Image Recognition using Convolution Network Layer
# Author Priyanka Rao
# Date 10/18/2016
################

from __future__ import division, print_function, absolute_import

from skimage import color, io
from scipy.misc import imresize
import numpy as np
from sklearn.cross_validation import train_test_split
import os
from glob import glob

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

###################################
# Import images
# All images are stored in data/t1 folder
###################################

files_path = '../data/t1/'

xImage_files_path = os.path.join(files_path, 'xImage*.jpg')
oImage_files_path = os.path.join(files_path, 'oImage*.jpg')

xImage_files = sorted(glob(xImage_files_path))
oImage_files = sorted(glob(oImage_files_path))

n_files = len(xImage_files) + len(oImage_files)
print(n_files)

size_image = 64

allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
ally = np.zeros(n_files)
count = 0
for fx in xImage_files:
    try:
        img = io.imread(fx)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1
    except:
        continue

for fo in oImage_files:
    try:
        img = io.imread(fo)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1
    except:
        continue

###################################
# Prepare train & test samples
###################################

# test-train split
X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)

# encode the Ys
Y = to_categorical(Y, 2)
Y_test = to_categorical(Y_test, 2)


###################################
# Image transformations
###################################

# normalisation of images
# image is normalized to zero center and scaled according to specified standard deviation.
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

###################################
# Define network architecture
###################################

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 64, 64, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)

# 3: Convolution layer with 64 filters
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

# 4: Convolution layer with 64 filters
conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')

# 5: Max pooling layer
network = max_pool_2d(conv_3, 2)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 512, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 8: Fully-connected layer with two outputs
network = fully_connected(network, 2, activation='softmax')

# Configure the network, I have used Stochastic Gradient Descent(SGD) optimizer to train the data
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='SGD',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model_xImage_oImage_6.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='../data/tmp/tflearn_logs/')
###########################################
      #OUTPUT
#After model is trained, event files are generated as the output file.
#These files are stored under the directory ../data/tmp/tflearn_logs/'
#TensorBoard uses these event files to visualize the data.
###################################

###################################
# Train model for 100 epochs
#The process of forward pass, loss function, backward pass, and parameter update is generally called one epoch
###################################
model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500,
      n_epoch=100, run_id='model_xImage_oImage_6', show_metric=True)

model.save('model_xImage_oImage_6_final.tflearn')

# this saved model can be used in future to recognize new images.
