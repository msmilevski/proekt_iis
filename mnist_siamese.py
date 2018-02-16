from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import h5py
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop, Adam
from keras import backend as K
import data_generator as generator
import time


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# This function creates a siamese network and trains it
# Train_file_location is the location of the training file, generated from alexnet.py
def start_traing_siamese_network(train_file_location="data/train_image_embeddings.hdf5", learning_rate=0.0001,
                                 epochs=50,
                                 batch_size=300, model_saving_location="new_model.h5"):
    # Opening file used for training the model
    training_file = h5py.File(train_file_location, 'r')
    image_embeddings_train = training_file["image_embeddings"]
    similarity_train = training_file["similarity"]

    # This is optional, it is not needed
    # valid_file = h5py.File("data/valid_image_embeddings.hdf5", 'r')
    # image_embeddings_valid = valid_file["image_embeddings"]
    # similarity_valid = valid_file["similarity"]

    input_shape = (1, 4096)
    # Network definition
    base_network = create_base_network(input_shape)
    # Input definition
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Merging the output of the two  siamese networks
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    # train
    rms = RMSprop(lr=learning_rate)
    steps_per_epoch = (len(image_embeddings_train) // batch_size) + 1
    t = time.time()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    model.fit_generator(generator.data_generator(image_embeddings_train, similarity_train, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=epochs)
    print(time.time() - t)
    # Saving model to location
    model.save(model_saving_location)
    # compute final accuracy on training and test sets
    # y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    # tr_acc = compute_accuracy(tr_y, y_pred)
    # y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    # te_acc = compute_accuracy(te_y, y_pred)
    #
    # print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    # print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))