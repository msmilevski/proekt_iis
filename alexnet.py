import sys
import os
import numpy as np
import time
import urllib
from numpy import random
import cv2

import tensorflow as tf
import h5py


# Creates convolutional layers
# if group is 2 it creates a split convolutional layer
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


# This function build the alexnet convnet and run our file through it, so it can extract the image embeddings
# The function also saves the embeddings, along with its similarities in a file, which is then used in the data_generator script
# File_name can be train, valid or test
# Image_directory_path is the location to your images
# Batch_size is number of images we are putting in the network
def get_embeddings_from_alexnet(file_name, image_directory_path='../images/', batch_size=256):
    train_x = np.zeros((1, 227, 227, 3)).astype(np.float32)
    train_y = np.zeros((1, 1000))
    xdim = train_x.shape[1:]
    ydim = train_y.shape[1]

    net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

    x = tf.placeholder(tf.float32, (None,) + xdim)

    # conv1
    # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11;
    k_w = 11;
    c_o = 96;
    s_h = 4;
    s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool1
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv2
    # conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5;
    k_w = 5;
    c_o = 256;
    s_h = 1;
    s_w = 1;
    group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool2
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv3
    # conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3;
    k_w = 3;
    c_o = 384;
    s_h = 1;
    s_w = 1;
    group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    # conv4
    # conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3;
    k_w = 3;
    c_o = 384;
    s_h = 1;
    s_w = 1;
    group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    # conv5
    # conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3;
    k_w = 3;
    c_o = 256;
    s_h = 1;
    s_w = 1;
    group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    # maxpool5
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # fc6
    # fc(4096, name='fc6')
    fc6W = tf.Variable(net_data["fc6"][0])
    fc6b = tf.Variable(net_data["fc6"][1])
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    # fc7
    # fc(4096, name='fc7')
    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    # fc8
    # fc(1000, relu=False, name='fc8')
    # fc8W = tf.Variable(net_data["fc8"][0])
    # fc8b = tf.Variable(net_data["fc8"][1])
    # fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

    # prob
    # softmax(name='prob'))
    # prob = tf.nn.softmax(fc8)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Defining file names and directory of the images
    file_path = file_name + '-data.npy'
    t1 = time.time()

    # Loading our data file
    data = np.load(file_path)
    print("Data:", data.shape)

    num_samples = len(data)

    # Creating placeholders for our batch
    images = []
    similarity = np.zeros(num_samples)
    image_embeddings = np.zeros((num_samples * 2, 4096))

    start = 0
    new_data = []
    for i in range(num_samples):
        print("Sample: ", i)
        sample = data[i]
        image_a_name, image_b_name = sample[0]
        similarity[i] = sample[1]

        image_a_name = image_directory_path + image_a_name
        image_b_name = image_directory_path + image_b_name
        # Reading images
        image_a = cv2.imread(image_a_name)
        image_b = cv2.imread(image_b_name)

        #     if type(image_a) is type(None) or type(image_b) is type(None):
        #
        #         continue
        #     else:
        #         new_data.append(data[i])
        #
        # np.save("test-data", new_data)

        # Pre-processing them before they enter the network
        # This is needed because these transformations were used during the training of the network
        image_a = image_a.astype('float32')
        image_a = cv2.resize(image_a, (227, 227))
        image_b = image_b.astype('float32')
        image_b = cv2.resize(image_b, (227, 227))
        image_a = preprocess_input(image_a)
        image_b = preprocess_input(image_b)
        # Filling our batch
        images.append(image_a)
        images.append(image_b)

        # Checking batch limit
        if i > 0 and ((i + 1) % batch_size == 0 or i == num_samples - 1):
            t = time.time()

            # Getting our image embeddings
            batch_embeddings = sess.run(fc7, feed_dict={x: images})
            print(batch_embeddings.shape)
            # Updating parameters
            images = []
            end = start + batch_size * 2
            image_embeddings[start:end, :] = batch_embeddings
            start = end
            print((time.time() - t) / 60.0)

    print((time.time() - t1) / 60.0)
    print(image_embeddings.shape)
    # Saving data in file
    save_name = "data/"+file_name+"_image_embeddings.hdf5"
    file = h5py.File(save_name, 'w')
    file.create_dataset("image_embeddings", data=image_embeddings)
    file.create_dataset("similarity", data=similarity)
    file.close()