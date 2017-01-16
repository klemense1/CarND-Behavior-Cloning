

import csv
import os
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

if __name__ == "__main__":
    DRIVING_LOG_PATH = '/Users/Klemens/Udacity_Nano_Car/P3_BehaviorCloning/driving_log.csv'
    IMG_DIR_PATH = '/Users/Klemens/Udacity_Nano_Car/P3_BehaviorCloning/IMG/'

    list_dir = os.listdir(IMG_DIR_PATH)
    driving_log = read_driving_log(DRIVING_LOG_PATH)

    for row in driving_log:
        img_from_log = row['front'].split('/')[-1]
        if img_from_log not in list_dir:
            print(img_from_log, 'not found')


    image_path = driving_log[0]['front']
    image = mpimg.imread(image_path)
    print('Image shape:', image.shape)
    print('Number of frames:', len(driving_log))

    steering_angle = np.array([float(log['angle']) for log in driving_log])

    plt.hist(steering_angle, bins=20)
    plt.title('Histogram showing distribution of active steering in dataset')
    plt.xlabel('steering angle [deg]')
    plt.ylabel('number of frames')
    plt.show()
