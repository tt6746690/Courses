#!/usr/bin/env python
# CSC320 Winter 2019 - Week 4
# 
# Code for plotting an image (must be greyscale, i.e. single channel PNG!!) as a surface
# Usage: python plotimageassurface <imagefilename>
#
# requirements: 
# pip3 install opencv-python numpy matplotlib sklearn
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mp_img


def plotimageassurface(image):
        # create the x and y coordinate arrays (here we just use pixel indices)
        xx, yy = np.mgrid[0:image.shape[0], 0:image.shape[1]]

        # create the figure
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, image ,rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)

        # show it
        plt.show()

def main():
        parser = argparse.ArgumentParser(description='Plots image as 3D surface')
        parser.add_argument('imagefilename', help='single channel 8-bit greyscale image filename')
        args = parser.parse_args()

        # read in the image from filename provided as first argument
        image = mp_img.imread(args.imagefilename)

        assert len(image.shape) == 2, "Error: input must be single-channel greyscale image!"

        plotimageassurface(image)

if __name__ == '__main__':
        main()