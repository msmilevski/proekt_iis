import numpy as np
import h5py
from keras.models import Model, load_model
import data_generator as generator
from keras import backend as K


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


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
num_samples = len(image_embeddings_valid)

model = load_model('model_1.h5', custom_objects={'contrastive_loss': contrastive_loss})
print(num_samples)

i = 0
valid_data = []
valid_y = []
for batch in generator.data_generator(image_embeddings_valid, similarity_valid, batch_size=1000):
    valid_data = batch[0]
    valid_y = batch[1]
    print("putting data", i)
    print(np.array(valid_data).shape)
    print("Predicting")
    y_pred = model.predict(valid_data)
    print("Computing accuracy")
    te_acc = compute_accuracy(valid_y, y_pred)

    # print('* Accuracy on training set: %0.2f%%' % (100 * te_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


    i += 1
    if i == 4:
        break

#valid_y =np.array(valid_y)
#valid_data = np.array(valid_data)

