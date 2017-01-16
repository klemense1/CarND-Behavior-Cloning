from datetime import timedelta

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy
import csv
import pytest
import os
import math
from skimage.color import rgb2gray, gray2rgb

from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Conv2D, Flatten, Dropout, MaxPooling2D, Lambda
from keras.utils import np_utils
from keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

NUM_CLASSES = 20

NUM_CHANNELS = 3

IMAGE_LENGTH_X = 80
IMAGE_LENGTH_Y = 160

IMAGE_SIZE = (IMAGE_LENGTH_X, IMAGE_LENGTH_Y)
IMAGE_SHAPE = (IMAGE_LENGTH_X, IMAGE_LENGTH_Y, NUM_CHANNELS)

LEARNING_RATE = 1e-3

PROJECT_DIR = '/Users/Klemens/Udacity_Nano_Car/P3_BehaviorCloning/'
UNITTEST_LOG_PATH = os.path.join(PROJECT_DIR, 'driving_log_unittest.csv')
DRIVING_LOG_PATH = os.path.join(PROJECT_DIR, 'driving_log.csv')


def load_image(image_path, resize_shape=None):
    """
    loads image and resizes it to defined

    Input:
    - image_path ... Full path to image

    Returns:
    - resized image as a numpy array
    """
    # numpy array will come as float. as imshow results in strange behaviour,
    # with float, convert it back to unsigned int
    # X_train_gen = np.uint8()
    img = mpimg.imread(image_path)

    if resize_shape is None:
        img_resized = img
    else:
        img_resized = scipy.misc.imresize(img, resize_shape)

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
    fields = ['front', 'left', 'right', 'angle', 'dunno1', 'dunno2', 'speed']
    with open(log_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=fields)
        for row in reader:
            driving_frame = {
                             'front': row['front'],
                             'angle': row['angle'],
                             'speed': row['speed']
                            }
            driving_log.append(driving_frame)

    return driving_log


def classes_to_angles(classes):
    """
    converts classes to angles

    Parameter:
    classes ... numpy array with classes [0, 1, ... NUM_CLASSES]

    Returns:
    angles ... numpy array with steering angles [-1; 1]
    """
    angles = classes * 2 / NUM_CLASSES - 1

    return angles


def angles_to_classes(angles_input):
    """
    converts angles to classes

    Parameter:
    angles ... numpy array with steering angles [-1; 1]

    Returns:
    classes ... numpy array with classes [0, 1, ... NUM_CLASSES]
    """
    angles_rounded = np.around(angles_input, 1)

    classes = (angles_rounded + 1) * NUM_CLASSES / 2

    return classes


def plot_image_and_angle(axis, img_path, angl):
    """
    plots image with corresponding steering angle in given matplotlib axis
    """
    img = load_image(img_path)

    axis.imshow(img)
    plt.setp(axis.get_xticklabels(), visible=False)
    plt.setp(axis.get_yticklabels(), visible=False)

    title = 'alpha = {} \nshape = {}'.format(angl, img.shape)

    axis.set_title(title)


def visualize_training_data(log_path):
    """
    visualizes training data using three images and corresponding angles
    """
    log_list = read_driving_log(log_path)

    steering_angle = np.array([float(log['angle']) for log in log_list])

    speed = np.array([float(log['speed']) for log in log_list])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    image_path = log_list[0]['front']
    angle = log_list[0]['angle']
    plot_image_and_angle(ax1, image_path, angle)

    image_path = log_list[1]['front']
    angle = log_list[1]['angle']
    plot_image_and_angle(ax2, image_path, angle)

    image_path = log_list[2]['front']
    angle = log_list[2]['angle']
    plot_image_and_angle(ax3, image_path, angle)

    fig.tight_layout()
    plt.show()


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
        image_path = row['front']
        image = load_image(image_path, IMAGE_SHAPE)
        img_list.append(image)

    features_sample = np.array(img_list).astype('float32')

    steering_angle = np.array([float(log['angle']) for log in log_list])

    classes = angles_to_classes(steering_angle)

    classes_sample = classes[:num_samples]

    labels_sample = np_utils.to_categorical(classes_sample, NUM_CLASSES)

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
            # print('Replicator:', int(batch_size/len(aslice)))
            # print('len(aslice)', len(aslice))
            # print('len(aslice_rep)', len(aslice_rep))
            # print('type(aslice)', type(aslice))
            # print('aslice', aslice)
            # print('aslice_rep', aslice_rep)

        assert len(aslice) == batch_size

        list_slices.append(aslice)

    return list_slices


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

        for list_slice in generate_equal_slices(shuffled_list, batch_size):
            assert len(list_slice) == batch_size

            img_list = []

            for row in list_slice:

                image_path = row['front']
                image = load_image(image_path, IMAGE_SHAPE)
                img_list.append(image)

            features_slice = np.array(img_list).astype('float32')

            steering_angle = np.array([float(log['angle']) for log in list_slice])
            classes = angles_to_classes(steering_angle)

            labels_slice = np_utils.to_categorical(classes, NUM_CLASSES)

            yield (features_slice, labels_slice)


def build_model():
    """
    Creates model using sequential call from keras

    Returns
    mdl ... keras model
    """
    mdl = Sequential()

    # normalization
    mdl.add(Lambda(lambda x: x/255. - 0.5, input_shape=IMAGE_SHAPE))

    # convolutions
    mdl.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode='same',))
    mdl.add((Dropout(0.5)))
    mdl.add(Activation('relu'))

    mdl.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode='same',))
    mdl.add((Dropout(0.5)))
    mdl.add(Activation('relu'))

    mdl.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode='same',))
    mdl.add((Dropout(0.5)))
    mdl.add(Activation('relu'))

    mdl.add(Conv2D(64, 3, 3, border_mode='same',))
    mdl.add((Dropout(0.5)))
    mdl.add(Activation('relu'))

    mdl.add(Conv2D(64, 3, 3, border_mode='same',))
    mdl.add((Dropout(0.5)))
    mdl.add(Activation('relu'))

    mdl.add(Flatten())

    mdl.add(Dense(128, activation='relu'))
    mdl.add(Dense(64, activation='relu'))
    mdl.add(Dense(NUM_CLASSES, activation='softmax'))

    # mdl.summary()

    return mdl


def calc_samples_per_epoch(epoch_size, batch_size):
    """
    Calcuates number of samples per epoch
    """
    spe = epoch_size + epoch_size % batch_size

    return spe


def train_model_generator(mdl, training_list, validation_list, epochs, batch_size):
    """
    trains model with fit_generator-command. Generator is used to fetch the
    input data.

    Parameters:
    mdl ...
    training_list ...
    validation_list ...
    epochs ...
    batch_size ...

    Returns:
    """
    spe_train = calc_samples_per_epoch(len(training_list), batch_size)
    spe_val = calc_samples_per_epoch(len(validation_list), batch_size)

    mdl.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(lr=LEARNING_RATE),
                metrics=['accuracy']
               )

    history = mdl.fit_generator(
                                my_generator(training_list, batch_size),
                                samples_per_epoch=spe_train,
                                nb_epoch=epochs
                               )

    score_eval = mdl.evaluate_generator(
                                   generator=my_generator(validation_list, batch_size),
                                   val_samples=spe_val,
                                  )

    accuracy_evaluation = score_eval[1]

    print("[Evaluation]%s: %.2f%%" % (mdl.metrics_names[1], accuracy_evaluation*100))

    return accuracy_evaluation


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
    mdl.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    mdl.fit(X_train,
            Y_train,
            batch_size=batch_size,
            nb_epoch=epochs,
            verbose=1)

    # evaluate the model
    score_eval = mdl.evaluate(X_val, Y_val, verbose=0)
    accuracy_evaluation = score_eval[1]

    print("[Evaluation]%s: %.2f%%" % (mdl.metrics_names[1], accuracy_evaluation*100))

    return accuracy_evaluation


def test_generate_slices():
    list_to_slice = [1, 2, 3, 4, 5]
    sliced_list_target = [[1, 2], [3, 4], [5, 5]]
    sliced_list_generated = generate_equal_slices(list_to_slice, batch_size=2)

    assert(sliced_list_target == sliced_list_generated)


def test_loading_image():

    log_list = read_driving_log(DRIVING_LOG_PATH)
    image_name = log_list[0]['front']

    test_shape = (32, 32)
    test_image = load_image(image_name, resize_shape=test_shape)
    assert test_image.shape[:2] == test_shape


def test_angle_to_class_generation():

    log_list = read_driving_log(DRIVING_LOG_PATH)
    angles = np.array([float(log['angle']) for log in log_list])

    classes = angles_to_classes(angles)

    angles_back = classes_to_angles(classes)

    classes_back = angles_to_classes(angles_back)

    assert np.equal(classes_back, classes).all()


def test_get_sample():

    test_num = 4
    features, labels = get_data_sample(DRIVING_LOG_PATH, test_num)

    shape_features = (test_num, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3)
    shape_labels = (test_num, NUM_CLASSES)

    assert(features.shape == shape_features and labels.shape == shape_labels)


def test_model_and_training_without_generator():

    labels, features = get_data_sample(UNITTEST_LOG_PATH, num_samples=None)

    model = build_model()

    acc_eval = train_model(model,
                           X_train=labels,
                           Y_train=features,
                           X_val=labels,
                           Y_val=features,
                           epochs=10,
                           batch_size=128)

    assert acc_eval > 0


if __name__ == "__main__":

    # visualize_training_data(DRIVING_LOG_PATH)

    model = build_model()

    log_list = read_driving_log(DRIVING_LOG_PATH)
    shuffled_list = shuffle(log_list)

    list_train, list_val = train_test_split(shuffled_list, test_size=0.2, random_state=42)

    print('Number of frames in Log File', len(log_list))
    print('Number of frames for training', len(list_train))
    print('Number of frames for validation', len(list_val))

    print('Samples per Epoche for Training', calc_samples_per_epoch(len(list_train), batch_size=128))
    print('Samples per Epoche for Validation', calc_samples_per_epoch(len(list_val), batch_size=128))

    train_model_generator(model, list_train, list_val, epochs=10, batch_size=128)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
