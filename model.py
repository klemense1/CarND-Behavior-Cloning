import numpy as np
import csv
import pytest
import os
import math
import cv2

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, MaxPooling2D, Lambda, Convolution2D

from keras.optimizers import Adam

import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import model_from_json
import random
import augmentation as augm


NUM_CHANNELS = 3

TRAIN_FROM_SCRATCH = True

RESIZE = True
NUM_IMAGE_MODES = 3 # for flipping

BATCH_SIZE = 90
NUM_EPOCHS = 5

LEARNING_RATE = 1e-4

if RESIZE:
    IMAGE_LENGTH_X = 160
    IMAGE_LENGTH_Y = 80
else:
    IMAGE_LENGTH_X = 320
    IMAGE_LENGTH_Y = 160

IMAGE_SIZE = (IMAGE_LENGTH_X, IMAGE_LENGTH_Y)
IMAGE_SHAPE = (IMAGE_LENGTH_Y, IMAGE_LENGTH_X, NUM_CHANNELS)

PROJECT_DIR = '/Users/Klemens/Udacity_Nano_Car/P3_BehaviorCloning/data/'
# IMAGE_DIR = os.path.join(PROJECT_DIR, 'data')
IMAGE_DIR = PROJECT_DIR
UNITTEST_LOG_PATH = os.path.join(PROJECT_DIR, 'driving_log_unittest.csv')
DRIVING_LOG_PATH = os.path.join(IMAGE_DIR, 'driving_log.csv')


def load_image(image_path, resize_shape=None):
    """
    loads image and resizes it to defined

    Input:
    - image_path ... Full path to image

    Returns:
    - resized image as a numpy array
    """

    if os.path.isfile(image_path):
        img = mpimg.imread(image_path)
    else:
        print('FILE NOT FOUND', image_path)

    if resize_shape is None:
        img_resized = img
    else:
        img_resized = cv2.resize(img, resize_shape)

    return img_resized


def read_driving_log(log_path):
    """
    reads driving log

    Input:
    - log_path ... Full path to log file

    Returns:
    - list of dictionaries with image paths and corresponding steering angle
    """

    driving_log = []
    fields = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    with open(log_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=fields, delimiter=',')
        for row in reader:
            center_frame = {'image': row['center'],
                            'steering': float(row['steering'])}

            if abs(float(row['steering']))>0.2:
                for i in range(30):
                    driving_log.append(center_frame)
            else:
                driving_log.append(center_frame)

            left_frame = {'image': row['left'][1:], # remove first whitespace
                          'steering': float(row['steering']) + 0.1}
            driving_log.append(left_frame)

            right_frame = {'image': row['right'][1:],  # remove first whitespace
                           'steering': float(row['steering']) - 0.1}
            driving_log.append(right_frame)

    return driving_log


def get_data_sample(log_path, num_samples):
    """
    reads driving log file and returns num_samples data samples

    Parameters:
    num_samples ... number of samples that shall be returned

    Returns:
    features_sample ... features with size = num_samples
    labels_sample ... label with size = num_samples
    """

    img_list = []

    log_list = read_driving_log(log_path)

    log_list_sample = log_list[:num_samples]

    for row in log_list_sample:
        image_path = row['center']

        if not os.path.isabs(image_path):
            image_path = os.path.join(IMAGE_DIR, image_path)

        image = load_image(image_path, IMAGE_SIZE)
        img_list.append(image)

    features_sample = np.array(img_list).astype('float32')

    steering_angle = np.array([float(log['steering']) for log in log_list_sample])

    labels_sample = steering_angle

    return features_sample, labels_sample


def generate_equal_slices(list_to_slice, batch_size):
    """
    Create slices of a list

    Parameter:
    list_to_slice ... list that shall be divided
    batch_size ...

    Return:
    list_slices ... slices of list
    """

    assert len(list_to_slice) > 1

    list_slices = []

    sample_size = len(list_to_slice)

    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        aslice = list_to_slice[start_i:end_i]
        if len(aslice) < batch_size:
            aslice_rep = aslice * math.ceil(batch_size/len(aslice))
            aslice = aslice_rep[:batch_size]

        assert len(aslice) == batch_size

        list_slices.append(aslice)

    return list_slices

def get_absolute_imgpath(image_path):
    """
    Checks if image_path is an absolute path and if not, changes that
    
    Parameters:
    image_path ... path to an image (absolute or not)
    
    Returns:
    absolute_image_path ... definitely absolute image path
    """
    if not os.path.isabs(image_path):
        # print(image_path, 'is not an absolute path...')
        absolute_image_path = os.path.join(IMAGE_DIR, image_path)
    else:
        # print(image_path, 'is an absolute path...')
        absolute_image_path = image_path

    return absolute_image_path



def augment_image_angle_pair(img_in, angle_in):
    """
    augments the image-angle pair using
    - image flipping
    - changing the brightness
    - slightly changing the angle
    
    Parameters:
    img_in ... raw image
    angle_in ... raw angle
    
    Returns:
    img_out ... augmentated image
    angle_out ... augmentated angle
    """
    flip_mode = random.randint(1, 2)
    shade_mode = random.randint(1,3)

    if flip_mode == 1:
        image_fl = img_in
        angle_fl = angle_in
    elif flip_mode == 2:
        image_fl, angle_fl = augm.flip_image_angle_pair(img_in, angle_in)

    if shade_mode == 1:
       img_out = image_fl
    elif shade_mode == 2:
       img_out = augm.change_brightness(image_fl)
    elif shade_mode == 3:
       img_out = augm.add_random_shadow(image_fl)

    angle_out = augm.gaussian_angle(angle_fl)

    return img_out, angle_out


def my_generator(log_file_list, batch_size):
    """
    Generator generating features and labels to have filled queue all the time
    while the model is training on the prior fetched samples.
    Yields features (image) and labels (classes generated from angles)

    Parameters:
    log_file_list ... list containing entries from log file
    batch_size ... batch size

    """

    while 1:

        shuffled_list = shuffle(log_file_list)

        for list_slice in generate_equal_slices(shuffled_list, int(batch_size)):

            img_list = []
            steering_angle_list = []

            for row in list_slice:
    
                img_path = get_absolute_imgpath(row['image'])
                image = load_image(img_path, IMAGE_SIZE)

                angle = row['steering']

                image, angle = augment_image_angle_pair(image, angle)

                img_list.append(image)
                steering_angle_list.append(angle)

            features_slice = np.array(img_list)#.astype('float32')

            labels_slice = np.array(steering_angle_list)

            assert features_slice.shape[0] == labels_slice.shape[0]

            yield ({'lambda_input_1': features_slice}, {'output': labels_slice})


def build_model():
    """
    Creates model using sequential call from keras

    Returns
    mdl ... keras model
    """
    mdl = Sequential()

    # normalization
    mdl.add(Lambda(lambda x: x/128. - 1, input_shape=IMAGE_SHAPE, name="input"))

    # trim image
    mdl.add(Lambda(lambda x: x[:, 10:-10, :, :]))

    # convolutions
    mdl.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='same',))
    mdl.add(MaxPooling2D(pool_size=(2, 2)))
    mdl.add((Dropout(0.5)))
    mdl.add(Activation('relu'))

    mdl.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='same',))
    mdl.add(MaxPooling2D(pool_size=(2, 2)))
    mdl.add((Dropout(0.5)))
    mdl.add(Activation('relu'))

    mdl.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='same',))
    mdl.add(MaxPooling2D(pool_size=(2, 2)))
    mdl.add((Dropout(0.5)))
    mdl.add(Activation('relu'))

    mdl.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same',))
    mdl.add((Dropout(0.5)))
    mdl.add(Activation('relu'))

    mdl.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same',))
    mdl.add((Dropout(0.5)))
    mdl.add(Activation('relu'))

    mdl.add(Flatten())

    mdl.add(Dense(128, activation='relu'))
    mdl.add(Dense(64, activation='relu'))
    mdl.add(Dense(1, name="output"))

    mdl.summary()

    return mdl


def calc_samples_per_epoch(epoch_size, batch_size):
    """
    Calcuates number of samples per epoch
    """
    spe = NUM_IMAGE_MODES * (epoch_size - epoch_size % batch_size + batch_size)

    return spe


def train_model_generator(mdl, training_list, validation_list, epochs, batch_size):
    """
    trains model with fit_generator-command. Generator is used to fetch the
    input data.

    Parameters:
    mdl ... keras model
    training_list ... list of dictionaries for each row read from csv
    validation_list ... list of dictionaries for each row read from csv
    epochs ... number of epochs
    batch_size ... batch size

    Returns:
    mdl ... keras model after training
    accuracy_evaluation ... accuracy of evaluation
    """
    spe_train = calc_samples_per_epoch(len(training_list), batch_size)
    spe_val = calc_samples_per_epoch(len(validation_list), batch_size)

    mdl.compile(
                loss='mean_squared_error',
                optimizer=Adam(),#lr=LEARNING_RATE
               )

    history = mdl.fit_generator(my_generator(training_list, batch_size),
                                samples_per_epoch=spe_train,
                                nb_epoch=epochs)

    score_eval = mdl.evaluate_generator(generator=my_generator(validation_list, batch_size),
                                        val_samples=spe_val)

    loss = score_eval

    print("[Evaluation]%s: %.2f%%" % (mdl.metrics_names, loss))

    return mdl, loss


def train_model(mdl, X_train, Y_train, X_val, Y_val, epochs, batch_size):
    """
    trains model with fit-command. Use for early testing of model

    Parameters:
    mdl ... created keras model
    X_train ... training features
    Y_train ... training labels
    X_val ... validation features
    Y_val ... validation labels
    epochs ... epochs

    Returns:
    score_eval ... Score of evaluation
    """
    mdl.compile(loss='mean_squared_error',
                optimizer='adam')

    mdl.fit(X_train,
            Y_train,
            batch_size=batch_size,
            nb_epoch=epochs,
            verbose=1)

    # evaluate the model
    score_eval = mdl.evaluate(X_val, Y_val, verbose=0)
    loss = score_eval

    print("[Evaluation]%s: %.2f%%" % (mdl.metrics_names, loss))

    return mdl, loss


def test_generate_slices():
    list_to_slice = [1, 2, 3, 4, 5]
    sliced_list_target = [[1, 2], [3, 4], [5, 5]]
    sliced_list_generated = generate_equal_slices(list_to_slice, batch_size=2)

    assert(sliced_list_target == sliced_list_generated)


def test_loading_image():

    log_list = read_driving_log(DRIVING_LOG_PATH)
    #image_name = log_list[0]['center']
    image_path = os.path.join(IMAGE_DIR, log_list[0]['center'])
    test_shape = (32, 32)
    test_image = load_image(image_path, resize_shape=test_shape)
    assert test_image.shape[:2] == test_shape


def test_get_sample():

    test_num = 4
    features, labels = get_data_sample(DRIVING_LOG_PATH, test_num)

    shape_features = (test_num, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3)
    shape_labels = (test_num, )

    assert(features.shape == shape_features and labels.shape == shape_labels)


def test_model_and_training_without_generator():

    labels, features = get_data_sample(UNITTEST_LOG_PATH, num_samples=None)

    model = build_model()

    _, acc_eval = train_model(model,
                              X_train=labels,
                              Y_train=features,
                              X_val=labels,
                              Y_val=features,
                              epochs=20,
                              batch_size=128)

    assert acc_eval > 0


if __name__ == "__main__":

    if TRAIN_FROM_SCRATCH:
        model = build_model()

    else:
        with open('model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)

        model.load_weights("model.h5")
        print("Loaded model from disk")


    log_list = read_driving_log(DRIVING_LOG_PATH)
    shuffled_list = shuffle(log_list)

    list_train, list_val = train_test_split(shuffled_list, test_size=0.2, random_state=41)

    print('Number of frames in Log File', len(log_list))
    print('Number of frames for training', len(list_train))
    print('Number of frames for validation', len(list_val))

    model, losses = train_model_generator(model,
                                       list_train,
                                       list_val,
                                       epochs=NUM_EPOCHS,
                                       batch_size=BATCH_SIZE)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model.h5")

    print("Saved model to disk")
