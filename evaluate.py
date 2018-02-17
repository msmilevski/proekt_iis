import numpy as np
import h5py
from keras.models import load_model
import data_generator as generator
import mnist_siamese as util


# Computes accuracy for predicted similarity probability
# model_path is the location of the trained model
# file_path is the location of the dataset we want to evaluate
# batch_size is the number of prediction we want to evaluate, because we can't load all image into memory
def evaluate_batch_on_file(model_path='model_1.h5', file_path="data/valid_image_embeddings.hdf5", batch_size=1000):
    valid_file = h5py.File(file_path, 'r')
    image_embeddings_valid = valid_file["image_embeddings"]
    similarity_valid = valid_file["similarity"]
    num_samples = len(image_embeddings_valid)

    model = load_model(model_path, custom_objects={'contrastive_loss': util.contrastive_loss()})
    print(num_samples)

    i = 0
    for batch in generator.data_generator(image_embeddings_valid, similarity_valid, batch_size=batch_size):
        valid_data = batch[0]
        valid_y = batch[1]
        print("putting data", i)
        print(np.array(valid_data).shape)
        print("Predicting")
        y_pred = model.predict(valid_data)
        print("Computing accuracy")
        te_acc = util.compute_accuracy(valid_y, y_pred)
        print('* Accuracy on set: %0.2f%%' % (100 * te_acc))

        i += 1
        if i == (num_samples // batch_size):
            break
