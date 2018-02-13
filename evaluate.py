import numpy as np
import h5py
from keras.models import Model, load_model
import data_generator as generator
from keras import backend as K

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

valid_file = h5py.File("data/valid_image_embeddings.hdf5", 'r')
image_embeddings_valid = valid_file["image_embeddings"]
similarity_valid = valid_file["similarity"]

model = load_model("model_1.h5")

for batch in generator.data_generator(image_embeddings_valid, similarity_valid, batch_size=2):
    images = batch[0]
    similarity = batch[1]
    prediction = model.predict(images)
    print(prediction)
    break
