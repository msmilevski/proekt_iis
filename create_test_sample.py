import numpy as np
from keras.models import load_model
import h5py
import cv2
import mnist_siamese as util


# Creates test samples for a given image
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


# Show the recommendations for an image
def show_images(model, indices, list_items, sample, item_name,
                directory='../images/'):
    similar_items = []
    image_embeddings = []
    print(len(sample))
    for idx in indices:
        # print(len(list_items))
        pair = list_items[idx][0]

        if item_name != pair[0]:
            item = pair[0]
            image_embeddings.append(sample[0][idx])
        else:
            item = pair[1]
            image_embeddings.append(sample[1][idx])
        # print(item)
        similar_items.append(item)

    # print(len(image_embeddings))
    # print(model.predict(
    #     [np.array(image_embeddings[1]).reshape(1, 1, 4096), np.array(image_embeddings[2]).reshape(1, 1, 4096)]))
    item_image = cv2.imread(directory + item_name)
    for item in similar_items:
        sim_item_image = cv2.imread(directory + item)
        item_image = np.concatenate((item_image, sim_item_image), axis=1)

    cv2.imshow("slika", item_image)
    cv2.waitKey(0)


# For given image creates a recommendation
# Image_name is path to the image we want to generate recommendations
# Model_name is path to the trained model
# Num_recommendations is the number of generated recommendations
def get_a_recommendation(image_name='1579.jpg', model_name='model_1.h5', num_recommendations=3):
    model = load_model(model_name, custom_objects={'contrastive_loss': util.contrastive_loss})
    item_name = image_name
    sample, list_items = create_test_sample(image_name=item_name)
    prediction = model.predict(sample)
    print(prediction)
    max_predictions = np.argsort(prediction.flatten())[-num_recommendations:]
    print(max_predictions)
    show_images(model, max_predictions, list_items, sample, item_name)

get_a_recommendation()