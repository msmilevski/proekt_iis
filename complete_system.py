import data_preprocessing as prepocessing
import alexnet as alexnet
import mnist_siamese as siamese
import create_test_sample as recommend
print("Downloading all the images")
prepocessing.save_images_from_file()
print("Creating training, valid and test data")
prepocessing.create_training_pairs()
print("Preparing for getting all the image embeddings from Alexnet")
alexnet.get_embeddings_from_alexnet('train')
alexnet.get_embeddings_from_alexnet('valid')
alexnet.get_embeddings_from_alexnet('test')
print("Preparing for training a siamese network")
siamese.start_traing_siamese_network()
print("Update image_name paramter to get recommendations")
recommend.get_a_recommendation()