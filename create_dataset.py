import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from tqdm import tqdm
import random
import pickle


def load_save_images(display=False, batches=10):
    image_location = 'facial_expressions-master/images'
    batch_size = len(os.listdir(image_location)) // batches
    index = 0
    for batch in range(batches):
        # TODO it misses the last few examples
        image_dict = {}
        lower_slice = batch_size*index
        upper_slice = batch_size*(index+1)
        index += 1
        if display:
            for image_name in tqdm(os.listdir(image_location)[lower_slice:upper_slice]):
                path = os.path.join(image_location, image_name)
                img1 = cv2.imread(path, flags=cv2.IMREAD_COLOR)
                res = cv2.resize(img1, dsize=(100, 100), interpolation=cv2.INTER_NEAREST)
                img2 = mpimg.imread(path)
                img_plot = plt.imshow(img2)
                img_plot.show()
                image_dict[image_name] = res[:, :, :1]
        else:
            for image_name in tqdm(os.listdir(image_location)[lower_slice:upper_slice]):
                p = os.path.join(image_location, image_name)
                img1 = cv2.imread(p, flags=cv2.IMREAD_COLOR)
                res = cv2.resize(img1, dsize=(100, 100), interpolation=cv2.INTER_NEAREST)
                image_dict[image_name] = res[:, :, :1]
        with open('./image_data_dict/images_' + str(batch)+'.dictionary', 'wb') as image_dict_file:
            pickle.dump(image_dict, image_dict_file)


def create_test_train(batches=10, cutoff=0.8):
    test_images = {}
    for batch in range(batches):
        with open('./image_data_dict/images_' + str(batch)+'.dictionary', 'rb') as image_dict_file:
            image_dict = pickle.load(image_dict_file)
        all_keys = list(image_dict.keys())
        cutoff_number = int(len(all_keys) * cutoff)
        random.shuffle(all_keys)
        train_images_keys = all_keys[:cutoff_number]
        test_image_keys = all_keys[cutoff_number:]
        train_images = {key: image_dict[key] for key in train_images_keys}
        with open('./image_data_dict/images_train' + str(batch) + '.dictionary', 'wb') as image_train_dict_file:
            pickle.dump(train_images, image_train_dict_file)
        test_images.update({key: image_dict[key] for key in test_image_keys})
    with open('./image_data_dict/images_test.dictionary', 'wb') as image_train_dict_file:
        pickle.dump(test_images, image_train_dict_file)


if __name__ == "__main__":
    load_save_images(display=False, batches=20)
    # create_test_train(batches=10)
    print("Done with creating dataset")
