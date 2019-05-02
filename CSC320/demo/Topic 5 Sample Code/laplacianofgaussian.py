#!/usr/bin/env python
# CSC320 Winter 2019 - Week 4
# 
# Code for calculating Laplacian of Gaussian based image edges

# NOTE: You require OpenCV2, numpy and matplotlib for this:
# 
# pip3 install opencv-python numpy matplotlib

import sys
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_image(image, description=''):
    img = plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(description)
    plt.show()

def gaussian_smooth(image, ksize=3):
        # This gives us a separable Gaussian filter of given size
        gaussian_filter = cv2.getGaussianKernel(ksize, -1)
        # Use function for convolution with separable filters (could also just use normal function to apply gaussian filter, and transpose)
        smoothed = cv2.sepFilter2D(image, -1, gaussian_filter, gaussian_filter.T)
        return smoothed

def laplacian(image):
        # The 3x3 Laplacian filter )
        laplacian_filter = np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]))

        # filter the image
        l_image = cv2.filter2D(image, -1, laplacian_filter)

        return l_image
    

def main():
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('filename', help='single channel 8-bit greyscale image filename')
        parser.add_argument('--gaussiansize', default=3, type=int, help='Gaussian filter size >=3')
        parser.add_argument('--edgethreshold', default=0.1, type=float, help='Threshold for edges')
        args = parser.parse_args()
        
        assert args.gaussiansize >= 3, "Gaussian filter size must be >=3"
        assert args.gaussiansize % 2 == 1, "Gaussian filter size should be odd"

        # read in the image from filename provided as first argument, convert to float32 representation
        image = cv2.imread(args.filename, cv2.IMREAD_GRAYSCALE)/255

        assert len(image.shape) == 2, "Error: input must be single-channel greyscale image!"

        # Smooth image using Gaussian filter
        smoothed = gaussian_smooth(image, args.gaussiansize)
        
        show_image(smoothed, 'Gaussian Smoothed Image')
        # norm_smoothed = cv2.normalize(smoothed, None, alpha=255, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # cv2.imwrite('log_gaussian_{}.png'.format(args.gaussiansize), norm_smoothed)

        # Calculate Laplacian of Gaussian
        LoG = laplacian(smoothed)
        # norm_LoG = cv2.normalize(LoG, None, alpha=255, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # cv2.imwrite('log_{}.png'.format(args.gaussiansize), norm_LoG)

        show_image(LoG, 'Laplacian of Gaussian')

        # Check for zero crossings
        # Look at your four neighbors, left, right, up and down
        # If they all have the same sign as you, then you are not a zero crossing
        # Else, if you have the smallest absolute value compared to your neighbors with opposite sign, then you are a zero crossing
        
        # This is a neat fast way of doing this
        # https://stackoverflow.com/questions/25105916/laplacian-of-gaussian-in-opencv-how-to-find-zero-crossings
        minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, np.ones((3,3)))
        maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((3,3)))
        zero_cross = np.logical_and(np.logical_or(np.logical_and(minLoG < 0,  LoG > 0), np.logical_and(maxLoG > 0, LoG < 0)), np.abs(LoG) > args.edgethreshold)

        show_image(zero_cross, 'Laplacian of Gaussian (Thresholded)')
        # norm_edges = zero_cross*255
        # cv2.imwrite('log_{}_{}.png'.format(args.gaussiansize, args.edgethreshold), norm_edges)
        
if __name__ == '__main__':
        main()