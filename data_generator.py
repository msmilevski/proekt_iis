import numpy as np

def data_generator(img_embed, similarity, batch_size):
    num_train_samples = len(img_embed)
    num_similarity = len(similarity)

    while 1:
        start_x = 0
        start_y = 0
        while start_x < num_train_samples and start_y < num_similarity:
            train_slice_x = img_embed[start_x: start_x + batch_size]
            train_slice_y = similarity[start_y: start_y + (batch_size // 2)]
            start_x += batch_size
            start_y += (batch_size // 2)
            train_slice_a = []
            train_slice_b = []

            for j in range(0, len(train_slice_x), 2):
                train_slice_a.append(train_slice_x[j])
                train_slice_b.append(train_slice_x[j + 1])

            train_slice_a = np.array(train_slice_a)
            train_slice_b = np.array(train_slice_b)
            shape = (train_slice_a.shape[0], 1, train_slice_a.shape[1])
            train_slice_a = train_slice_a.reshape(shape)
            train_slice_b = train_slice_b.reshape(shape)

            #print(train_slice_y.shape)
            yield ([train_slice_a, train_slice_b], train_slice_y)