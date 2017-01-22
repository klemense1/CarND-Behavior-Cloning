#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:10:30 2017

@author: Klemens
"""
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def flip_image_angle_pair(img_in, angle_in):
    """
    flip image and angle pair
    
    Parameters:    
    img_in
    angle_in

    Returns:
    image
    angle
    """
    image = cv2.flip(img_in, 1)
    angle = -angle_in

    return image, angle


def add_random_shadow(image):
    """
    adds randomly placed shadowing to the image
    
    Parameters:
    image ... image to be shaded
    
    Returns:
    image_shade ...image with placed shadow
    
    """
    Y_TOP = 0
    X_BOTTOM = image.shape[0]
    X_TOP = image.shape[1]*np.random.uniform()
    X_BOT = image.shape[1]*np.random.uniform()

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    shadow_mask = np.zeros(hsv[:,:,1].shape)

    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-Y_TOP) * (X_BOT-X_TOP) - (X_BOTTOM - Y_TOP) * (Y_m-X_TOP) >=0)]=1

    lightness_change = 0.5

    pos_mask = shadow_mask==1
    neg_mask = shadow_mask==0
   
    if np.random.randint(2)==1:
        hsv[:,:,2][pos_mask] = hsv[:,:,2][pos_mask] * lightness_change
    else:
        hsv[:,:,2][neg_mask] = hsv[:,:,2][neg_mask] * lightness_change

    hsv[:, :, 2][hsv[:, :, 2]>255] = 255

    image_shade = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    return image_shade


def change_brightness(image):
    """
    changes brightness of picture
    
    Parameters:
    image ... input image
    
    Returns:
    img_brightness ... image with different brightness
    """
    lightness_change = random.uniform(0.6, 1.2)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hsv[:, :, 2] = hsv[:, :, 2] * lightness_change
    hsv[:, :, 2][hsv[:, :, 2]>255] = 255

    img_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return img_brightness



def gaussian_angle(ang):
    """
    returns a randomly chosen angle according to a uniform distribution 
    around the given ang
    
    Parameters:
    ang ... given angle
    
    Returns:
    new_ang ... new angle
    """
    new_ang = ang * (1.0 + np.random.uniform(-1, 1)/25)
    return new_ang


def visualize_training_data(img_in, ang_in):
    """
    visualizes training data using three images and corresponding angles
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    # RAW
    img = img_in
    ang = ang_in

    ax = axes[0,0]
    ax.imshow(img)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    title = 'alpha = {} \nshape = {}'.format(ang, img.shape)
    ax.set_title(title)


    img = change_brightness(img_in)
    ang = ang_in

    ax = axes[0,1]
    ax.imshow(img)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    title = 'Image shadowing \n alpha = {}'.format(ang)
    ax.set_title(title)
    

    img = add_random_shadow(img_in)
    ang = ang_in

    ax = axes[1,0]
    ax.imshow(img)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    title = 'Random Shadowing \n alpha = {}'.format(ang)
    ax.set_title(title)

    
    img = img_in
    ang = gaussian_angle(ang_in)

    ax = axes[1,1]
    ax.imshow(img)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    title = 'Gaussian Angle \n alpha = {}'.format(ang)
    ax.set_title(title)

    plt.show()


if __name__ == "__main__":
    img_path = '/Users/Klemens/Udacity_Nano_Car/P3_BehaviorCloning/IMG_UNITTEST/center_2017_01_08_11_31_42_704.jpg'
    test_image = mpimg.imread(img_path)
    test_angle = 0.1

    visualize_training_data(test_image, test_angle)