

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
    DRIVING_LOG_PATH = '/Users/Klemens/Udacity_Nano_Car/P3_BehaviorCloning/170121_data/driving_log.csv'
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

    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    ax1.hist(steering_angle, bins=20)
    ax1.set_title('Histogram showing distribution of active steering in dataset')
    ax1.set_xlabel('steering angle [deg]')
    ax1.set_ylabel('number of frames')

    ax2.plot(steering_angle)
    ax2.set_title('Steering over time')
    ax2.set_xlabel('frames')
    ax2.set_ylabel('steering angle [deg]')

    fig.tight_layout()
    plt.show()
