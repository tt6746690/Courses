#!/usr/bin/env python
# CSC320 Winter 2019 - Week 4
# 
# Code for calculating Difference of Gaussian based image edges

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

def difference_of_gaussian(image, ksize1, ksize2):
        assert ksize1 < ksize2, "ksize2 must be greater than ksize1"

        gaussian1 = gaussian_smooth(image, ksize1)
        gaussian2 = gaussian_smooth(image, ksize2)

        return gaussian2-gaussian1
    

def main():
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('filename', help='single channel 8-bit greyscale image filename')
        parser.add_argument('--gaussiansizelow', default=3, type=int, help='Smaller Gaussian filter size >=3')
        parser.add_argument('--gaussiansizehigh', default=5, type=int, help='Larger Gaussian filter size >=3')
        parser.add_argument('--edgethreshold', default=0.1, type=float, help='Threshold for edges')
        args = parser.parse_args()
        
        assert args.gaussiansizelow >= 3 and args.gaussiansizehigh >= args.gaussiansizelow, "Gaussian filter size must be >=3"
        assert args.gaussiansizelow % 2 == 1 and args.gaussiansizehigh % 2 == 1, "Gaussian filter size should be odd"

        # read in the image from filename provided as first argument, convert to float32 representation
        image = cv2.imread(args.filename, cv2.IMREAD_GRAYSCALE)/255

        assert len(image.shape) == 2, "Error: input must be single-channel greyscale image!"

        # Calculate Difference of Gaussian
        DoG = difference_of_gaussian(image, args.gaussiansizelow, args.gaussiansizehigh)
        norm_DoG = cv2.normalize(DoG, None, alpha=255, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cv2.imwrite('log_{}_{}.png'.format(args.gaussiansizelow, args.gaussiansizehigh), norm_DoG)

        show_image(DoG, 'Difference of Gaussian')

        edges = DoG > args.edgethreshold
        show_image(edges, 'Difference of Gaussian (Thresholded)')
        norm_edges = edges*255
        cv2.imwrite('dog_{}_{}_{}.png'.format(args.gaussiansizelow, args.gaussiansizehigh, args.edgethreshold), norm_edges)

         # This is a neat fast way of doing this
        # https://stackoverflow.com/questions/25105916/laplacian-of-gaussian-in-opencv-how-to-find-zero-crossings
        minDoG = cv2.morphologyEx(DoG, cv2.MORPH_ERODE, np.ones((3,3)))
        maxDoG = cv2.morphologyEx(DoG, cv2.MORPH_DILATE, np.ones((3,3)))
        zero_cross = np.logical_and(np.logical_or(np.logical_and(minDoG < 0,  DoG > 0), np.logical_and(maxDoG > 0, DoG < 0)), np.abs(DoG) > args.edgethreshold)

        show_image(zero_cross, 'Difference of Gaussian Zero-Crossings (Thresholded)')
        norm_edges = zero_cross*255
        cv2.imwrite('dog_zerocross_{}_{}_{}.png'.format(args.gaussiansizelow, args.gaussiansizehigh, args.edgethreshold), norm_edges)
        
if __name__ == '__main__':
        main()