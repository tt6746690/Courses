#!/usr/bin/env python
# CSC320 Winter 2019 - Week 4
# 
# Code for calculating a very simple finite differences based gradient image

# NOTE: You require OpenCV2, numpy and matplotlib for this:
# 
# pip3 install opencv-python numpy matplotlib

import sys
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt


def write_image(image, filename):
        norm = cv2.normalize(image, None, alpha=255, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cv2.imwrite(filename, norm)


def show_image(image, description=''):
    img = plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(description)
    plt.show()


def gradient(image, type='sobel'):
        # Creates a 2D convolutional filter (see https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html )
        # This is the same as performing for each pixel the finite difference calculation given in class
        if type == 'forwarddiff':
                filter_x = np.array(([0, 0, 0], [0, -1, 1], [0, 0, 0]))
                filter_y = filter_x.T
        if type == 'backwarddiff':
                filter_x = np.array(([0, 0, 0], [-1, 1, 0], [0, 0, 0]))
                filter_y = filter_x.T
        elif type == 'centraldiff':
                filter_x = np.array(([0, 0, 0], [-1, 0, 1], [0, 0, 0]))/2
                filter_y = filter_x.T
        elif type == 'smooth':
                filter_x = np.array(([0, 0, 0], [1, 2, 1], [0, 0, 0]))/4
                filter_y = filter_x.T
        else:
                filter_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
                filter_x = filter_y.T

        # filter the image
        grad_x = cv2.filter2D(image, -1, filter_x)
        grad_y = cv2.filter2D(image, -1, filter_y)

        return grad_x, grad_y


def polar(grad_x, grad_y):
    magnitude = np.sqrt(grad_x**2+grad_y**2)
    theta = np.arctan2(grad_y, grad_x)

    theta = theta/np.pi

    return magnitude.astype(np.float32), theta.astype(np.float32)
    

def main():
        parser = argparse.ArgumentParser(description='CSC320 Gradient Filters')
        parser.add_argument('filename', help='single channel 8-bit greyscale image filename')
        parser.add_argument('--filter', default='sobel', choices=['sobel', 'smooth', 'forwarddiff', 'backwarddiff', 'centraldiff'], help='gradient filter type')
        parser.add_argument('--edgethreshold', default=0.1, type=float, help='Threshold for edges')
        args = parser.parse_args()

        # read in the image from filename provided as first argument, convert to float32 representation
        image = cv2.imread(args.filename, cv2.IMREAD_GRAYSCALE)/255

        assert len(image.shape) == 2, "Error: input must be single-channel greyscale image!"

        # calculate the x/y gradient images using the given filter
        (grad_x, grad_y) = gradient(image, args.filter)

        show_image(grad_x, 'df/dx')
        show_image(grad_y, 'df/dy')
        
        if args.filter == 'smooth':
                smooth = grad_x + grad_y
                show_image(smooth, 'smoothed image')
        else:
                # calculate the gradient magnitude/angle images using the given filter
                (magnitude, theta) = polar(grad_x, grad_y)

                show_image(magnitude, 'gradient magnitude')
                write_image(magnitude, 'gradient_magnitude.png')

                show_image(theta, 'gradient angle')
                write_image(theta, 'gradient_angle.png')

                # show edges given threshold
                show_image(magnitude >= args.edgethreshold, 'gradient thresholded')
                write_image((magnitude >= args.edgethreshold)*255, 'gradient_threshold_{}.png'.format(args.edgethreshold))
                

if __name__ == '__main__':
        main()