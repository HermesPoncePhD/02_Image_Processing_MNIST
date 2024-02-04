## IMAGE PROCESSING USING MNIST DATABSE ##
# Created by: Hermes Ponce
#===================================================================
# Description:
# High-level tutorial into Deep Learning using MNIST database and 
# TensorFlow Library. The main goal is to identify handwritten images.

#===================================================================
from IPython import get_ipython
get_ipython().magic('reset -sf')

# Libraries and settings
import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

# Settings
LEARNING_RATE = 1e-4
# Set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 2500        
    
DROPOUT = 0.5
BATCH_SIZE = 50

# Set to 0 to train on all available data
VALIDATION_SIZE = 2000

# Image number to output
IMAGE_TO_DISPLAY = 10

# DATA PREPARATION:
# Read training data from CSV file 
data = pd.read_csv('train.csv')

print('data({0[0]},{0[1]})'.format(data.shape))
print(data.head())

# Each row represents an image of a handwritten digit and a label with the value of this digit
# Every image is a stretched array of 785 pixels values (28x28 px)
images = data.iloc[:,1:].values
images = images.astype(np.float64) # Convert into floating points

# Convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
print('images({0[0]},{0[1]})'.format(images.shape))

image_size = images.shape[1]
print ('image_size => {0}'.format(image_size))
# In this case all images are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))

# To output one of the images, we reshape this long string of pixels into a 2-dimensional array, which is basically a grayscale image.
def display(img):
    # (784) => (28,28)
    one_image = img.reshape(image_width,image_height)
    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)

# If you want to change the colormaps (cmap='viridis')
from matplotlib import colormaps
list(colormaps)

# Output image     
display(images[IMAGE_TO_DISPLAY])

print('labels_flat({0})'.format(len(data['label'])))
print ('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY,data['label'][IMAGE_TO_DISPLAY]))

labels_count = np.unique(data['label']).shape[0]
print('labels_count => {0}'.format(labels_count))

# For most classification problems "one-hot vectors" are used.
# It is a vector that contains a single element equal to 1 and the rest of them equal to 0.
# In this case, the nth digit is represented as a zero vector with 1 in the nth position.
# Convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(data['label'], labels_count)
labels = labels.astype(np.uint8)
print('labels({0[0]},{0[1]})'.format(labels.shape))
print ('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))

# Split data into training & validation
# 1. Simple way is to select from 0 to 2000 for validation data and the remain for training
# But it is not correct due to the linear selection, it is not random
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]
train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]
print('train_images({0[0]},{0[1]})'.format(train_images.shape))
print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))

# 2. The proper way to do it the split data into train and test
# from sklearn.model_selection import train_test_split
# train_images, validation_images, train_labels, validation_labels = train_test_split(images, labels, test_size=0.1, random_state=42)
# train_images.shape
# validation_images.shape

# TensorFlow graph
# For this NN model, a lot of weights and biases are created.
# Generally, weights should be initalised with a small amount of noise for symmetry breaking, and to prevent 0 gradients.

# Weight and bias initialization
def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# For this problem we use zero padded convolutions so that the output is the same size as the input
# In this case, convolution layer is used to get the shape of each digit.
# It uses learnable kernels/filters each of which corresponds to one particular shape pattern.

# Convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Pooling is plain max pooling over 2x2 blocks.
# It is used for downsampling of the data.
# In this case, it splits the image into square 2-pixel blocks.
# Only keeps maximum value for each of those blocks (ie [[0,3],[4,2]] => 4 or [[0,1],[1,1]] => 1)

# Pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# The good thing about neural networks that nay NN can be used as a layer in a large multilayer NN,
# meaning that ouput of one can be used as input for another. 
# This sequential approach can create very sophisticated NN with multiple layers.
# They are also called Deep Neural Networks.

# In this case, we used two convolution layers with pooling in between them, 
# then densely connected layer followed by dropout and lastly readout layer.

# Input & output of NN
import tensorflow.compat.v1 as tf 
tf.compat.v1.disable_eager_execution() # Disabling eager mode
# Images
x = tf.compat.v1.placeholder('float',shape=[None,image_size])
# Labels
y_ = tf.compat.v1.placeholder('float', shape=[None,labels_count])

#================================
### First convolutional layer ###
#================================
# [Patch 5x5, 1 input channel, 32 output channels]
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# (37800,784) => (37800,28,28,1)
# 37800 images, 28x28 pixels, grayscale
image = tf.reshape(x, [-1,image_width,image_height,1])
print (image.get_shape()) # =>(37800,28,28,1)

h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
print (h_conv1.get_shape()) # => (37800, 28, 28, 32)
h_pool1 = max_pool_2x2(h_conv1)
print (h_pool1.get_shape()) # => (37800, 14, 14, 32)

# Prepare for visualization
# Display 32 fetures in 4 by 8 grid
layer1 = tf.reshape(h_conv1,(-1, image_height, image_width, 4 ,8))  
# Reorder so the channels are in the first dimension, x and y follow.
layer1 = tf.transpose(layer1, (0, 3, 1, 4,2))
layer1 = tf.reshape(layer1, (-1, image_height*4, image_width*8))

#=================================
### Second convolutional layer ###
#=================================
# [Patch 5x5, 32 input channel (from previous layer), 64 output channels]
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print (h_conv2.get_shape()) # => (37800, 14,14, 64)
h_pool2 = max_pool_2x2(h_conv2)
print (h_pool2.get_shape()) # => (37800, 7, 7, 64)

# Prepare for visualization
# Display 64 fetures in 4 by 16 grid
layer2 = tf.reshape(h_conv2, (-1, 14, 14, 4 ,16))  
# Reorder so the channels are in the first dimension, x and y follow.
layer2 = tf.transpose(layer2, (0, 3, 1, 4,2))
layer2 = tf.reshape(layer2, (-1, 14*4, 14*16)) 

# Now that the image size is reduced to 7x7, we add a fully-connected layer 
# with 1024 neurones to allow processing on the entire image (each of the neurons 
# of the fully connected layer is connected to all the activations/outpus of the previous layer)

#==============================
### Densely connected layer ###
#==============================
# [Patch 7x7, 64 input channel (from previous layer), 1024 output channels]
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
print (h_pool2_flat.get_shape()) # => (37800, 3136)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
print (h_fc1.get_shape()) # => (37800, 1024)

# To prevent overfitting, we apply droput before the readout layer.
# Dropout removes some nodes from the network at each training stage.

# Dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer for deep net
W_fc2 = weight_variable([1024, labels_count])
b_fc2 = bias_variable([labels_count])

y_ = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
print (y_.get_shape()) # => (37800, 10)

# To evaluate network performance we use cross-entropy and to minimise it ADAM optimiser is used.
# Cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y_))

# Optimisation function
train_step = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# Evaluation
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# To predict values from test data, highest probability is picked from "one-hot vector" 
# indicating that chances of an image being one of the digits are highest.

# Prediction function
#[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
predict = tf.argmax(y_,1)

#==================================
### Train, validate and predict ###
#==================================
# Ideally, we should use all data for every step of the training, 
# but that's expensive. So, instead, we use small "batches" of random data.
# This method is called stochastic training. It is cheaper, faster and gives much of the same result.

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# We create the function that serves data by batches
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # When all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # Finished epoch
        epochs_completed += 1
        # Shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # Start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

# Now when all operations for every variable are defined in TensorFlow graph 
# all computations will be performed outside Python environment.

#===============================
### Start TensorFlow session ###
#===============================
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

# Visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []
display_step = 1

for i in range(TRAINING_ITERATIONS):
    # Get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)        

    # Check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys, keep_prob: 1.0})       
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:BATCH_SIZE], 
                                                           y_: validation_labels[0:BATCH_SIZE], 
                                                            keep_prob: 1.0})                                  
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            validation_accuracies.append(validation_accuracy)
            
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
        
        # Increase display_step
        if i%(display_step*10) == 0 and i:
            display_step *= 10
    # Train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})

# After training is done, it's good to check accuracy on data that wasn't used in training
# Check final accuracy on validation set  
if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={x: validation_images,y_: validation_labels,keep_prob: 1.0})
    print('validation_accuracy => %.4f'%validation_accuracy)
    plt.plot(x_range, train_accuracies,'-b', label='Training')
    plt.plot(x_range, validation_accuracies,'-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax = 1.1, ymin = 0.7)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.show()

#======================================
### Read test data and check it out ###
#======================================
# Test data contains only images and labels are missing. 
# Otherwise, the structure is similar to training data

# Read test data from CSV file 
test_images = pd.read_csv('test.csv').values
test_images = test_images.astype(np.float64)

# Convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)
print('test_images({0[0]},{0[1]})'.format(test_images.shape))

# Predict test set
# Predicted_lables = predict.eval(feed_dict={x: test_images, keep_prob: 1.0})

# Using batches is more resource efficient
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], 
                                                                                keep_prob: 1.0})

print('predicted_lables({0})'.format(len(predicted_lables)))

# Output test image and prediction
display(test_images[IMAGE_TO_DISPLAY])
print ('predicted_lables[{0}] => {1}'.format(IMAGE_TO_DISPLAY,predicted_lables[IMAGE_TO_DISPLAY]))

# Save results
np.savetxt('submission_softmax.csv', 
            np.c_[range(1,len(test_images)+1),predicted_lables], 
            delimiter=',', 
            header = 'ImageId,Label', 
            comments = '', 
            fmt='%d')

layer1_grid = layer1.eval(feed_dict={x: test_images[IMAGE_TO_DISPLAY:IMAGE_TO_DISPLAY+1], keep_prob: 1.0})
plt.axis('off')
plt.imshow(layer1_grid[0], cmap=cm.seismic)

# To release resources held by the other sessions.
sess.close()








