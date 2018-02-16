import numpy as np
from keras import backend as K
from keras.models import Model, load_model
import h5py
import cv2


def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_test_sample(image_name='760.jpg'):
    train_data = np.load("train-data.npy")
    valid_data = np.load("valid-data.npy")
    test_data = np.load("test-data.npy")
    train_file = h5py.File("data/train_image_embeddings.hdf5", 'r')
    valid_file = h5py.File("data/valid_image_embeddings.hdf5", 'r')
    test_file = h5py.File("data/test_image_embeddings.hdf5", 'r')
    image_embeddings_train = train_file["image_embeddings"]
    image_embeddings_valid = valid_file["image_embeddings"]
    image_embeddings_test = test_file["image_embeddings"]

    test_image_a = []
    test_image_b = []
    list_items = []
    for i in range(len(train_data) - 1):

        if image_name in train_data[i][0]:
            list_items.append(train_data[i])
            test_image_a.append(image_embeddings_train[i])
            test_image_b.append(image_embeddings_train[i + 1])

    for i in range(len(valid_data) - 1):
        if image_name in valid_data[i][0]:
            list_items.append(valid_data[i])
            test_image_a.append(image_embeddings_valid[i])
            test_image_b.append(image_embeddings_valid[i + 1])

    # for i in range(len(test_data) - 1):
    #     if image_name in test_data[i][0]:
    #         list_items.append(test_data[i])
    #         test_image_a.append(image_embeddings_test[i])
    #         test_image_b.append(image_embeddings_test[i + 1])

    test_image_a = np.array(test_image_a)
    test_image_b = np.array(test_image_b)
    shape = (test_image_a.shape[0], 1, test_image_a.shape[1])
    test_image_a = test_image_a.reshape(shape)
    test_image_b = test_image_b.reshape(shape)
    return [test_image_a, test_image_b], list_items


def show_images(indices, list_items, item_name,
                directory='../images/'):
    similar_items = []
    for idx in indices:
        print(len(list_items))
        pair = list_items[idx][0]
        item = [s for s in pair if (s != item_name)]
        print(item[0])
        similar_items.append(item[0])

    item_image = cv2.imread(directory+item_name)
    for item in similar_items:
        sim_item_image = cv2.imread(directory+item)
        item_image = np.concatenate((item_image, sim_item_image), axis=1)

    cv2.imshow("slika", item_image)
    cv2.waitKey(0)




model = load_model('model_1.h5', custom_objects={'contrastive_loss': contrastive_loss})
item_name = '156.jpg'
sample, list_items = create_test_sample(image_name=item_name)
prediction = model.predict(sample)
print(prediction)
max_predictions = np.argsort(prediction.flatten())[-3:]
print(max_predictions)
show_images(max_predictions, list_items, item_name)
