from train_network import create_one_hot_encoding, encode_labels, create_labels
from keras.models import load_model, Sequential
import os
import pickle
import numpy as np
import tensorflow as tf

input_width = 128
input_height = 128


if __name__ == "__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    label_dict, emotions_list = create_labels()
    one_hot_encoding = create_one_hot_encoding(emotions_list)
    encoded_labels = encode_labels(label_dict, one_hot_encoding, emotions_list)

    with open('./image_data_dict/images_test.dictionary', 'rb') as image_dict_file:
        validate_image_dict = pickle.load(image_dict_file)
        amount_images = len(validate_image_dict.keys())
        x_train = np.empty(shape=(amount_images, input_width, input_height, 1), dtype=int)
        y_train = np.empty(shape=(amount_images, 8), dtype=int)
        index = 0
        for image_name, image_data in validate_image_dict.items():
            if image_name in encoded_labels.keys() and image_data.shape[0] == input_width:
                x_train[index] = image_data
                y_train[index] = (encoded_labels[image_name])
                index += 1

    model_path = "model"
    model_name = "facial_model.h5"
    path = os.path.join(model_path, model_name)
    model = load_model(path)
    model.fit(x_train, y_train, batch_size=32, epochs=1, validation_split=.1, verbose=2)
