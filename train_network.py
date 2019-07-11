from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.regularizers import l2
import tensorflow as tf
import numpy as np
import os
import pickle
import gc
from tqdm import tqdm
import random


def create_model():
    model = Sequential()
    num_features = 64
    num_labels = 8
    input_width = 100
    input_height = 100

    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(input_width, input_height, 1),
                     data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(2 * 2 * 2 * num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * 2 * num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * num_features, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])
    return model


def create_labels():
    label_location = 'facial_expressions-master/data/legend.csv'
    with open(label_location, mode='r') as labelFile:
        label_dict = {}
        next(labelFile)
        for label in labelFile:
            user_id, image_name, emotion = label.split(",")
            label_dict[image_name] = emotion[:-1].lower()

        emotions_list = []
        for key, value in label_dict.items():
            if value not in emotions_list:
                emotions_list.append(value)
        return label_dict, emotions_list


def create_one_hot_encoding(emotions_list):
    size_encoding = len(emotions_list)
    one_hot_encoding = np.empty(shape=(size_encoding, size_encoding))
    index = 0
    for label in emotions_list:
        one_hot_row = np.zeros(shape=(1, size_encoding))
        one_hot_row[0][index] = 1
        one_hot_encoding[index] = one_hot_row
        index += 1
    return one_hot_encoding


def encode_labels(labels, one_hot_encoding, emotions_list):
    encoded_labels = {}
    for image_name, emotion in labels.items():
        index = emotions_list.index(emotion)
        one_hot = one_hot_encoding[index]
        encoded_labels[image_name] = one_hot
    return encoded_labels


if __name__ == "__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    labelDict, emotions_list = create_labels()
    one_hot_encoding = create_one_hot_encoding(emotions_list)
    encoded_labels = encode_labels(labelDict, one_hot_encoding, emotions_list)

    model_path = "model"
    model_name = "facial_model.h5"
    path = os.path.join(model_path, model_name)
    if model_name not in os.listdir(model_path):
        model = create_model()
        model.save(path)
    else:
        model = load_model(path)

    order = list(range(20))
    random.shuffle(order)
    for run in range(5):
        for batch in tqdm(order):
            with open('./image_data_dict/images_' + str(batch) + '.dictionary', 'rb') as image_dict_file:
                train_image_dict = pickle.load(image_dict_file)
                amount_images = len(train_image_dict.keys())
                x_train = np.empty(shape=(amount_images, 100, 100, 1), dtype=int)
                y_train = np.empty(shape=(amount_images, 8), dtype=int)
                index = 0
                for image_name, image_data in train_image_dict.items():
                    if image_name in encoded_labels.keys() and image_data.shape[0] == 100:
                        x_train[index] = image_data
                        y_train[index] = (encoded_labels[image_name])
                        index += 1

                model.fit(x_train, y_train, batch_size=16, epochs=3, validation_split=.1, verbose=2)
                model.save(model_name)
                del train_image_dict, x_train, y_train
                image_dict_file.close()
                gc.collect()
    print("Done with run " + str(run))
