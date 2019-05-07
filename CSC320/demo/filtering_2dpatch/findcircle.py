#!/usr/bin/env python
# CSC320 Winter 2019 - Week 4
# 
# Code for finding RANSCAC circles

# NOTE: You require OpenCV2, numpy and matplotlib for this:
# 
# pip3 install opencv-python numpy matplotlib

import sys
import argparse
from math import sqrt, fabs

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

def draw_circle(image, circle, maxvote):
        red = int((circle.votes*255)/maxvote)
        # Note: for come reasons x/y coordinates for OpenCV circle are reversed!
        cv2.circle(image, (int(round(circle.b)), int(round(circle.a))), radius=int(round(circle.r)), color=(red, 0, 0), thickness=2, lineType=8)

def gaussian_smooth(image, ksize=3):
        # This gives us a separable Gaussian filter of given size
        gaussian_filter = cv2.getGaussianKernel(ksize, -1)
        # Use function for convolution with separable filters (could also just use normal function to apply gaussian filter, and transpose)
        smoothed = cv2.sepFilter2D(image, -1, gaussian_filter, gaussian_filter.T)
        return smoothed


def find_edges(image, threshold1, threshold2):
        # remove noise/low scale information
        smoothed = gaussian_smooth(image, ksize=9)

        # run Canny
        cannyedges = cv2.Canny(smoothed, threshold1, threshold2)
        return cannyedges


class circle:
        def __init__(self, a, b, r, votes):
                self.a = a
                self.b = b
                self.r = r
                self.votes = votes


def ransac_circle(edges, threshold_close, threshold_inlier, inlier_threshold, max_iterations):
        iterations = 0

        edge_indices = range(len(edges))
        candidates = []

        while iterations < max_iterations:
                iterations += 1

                # RANDOM SAMPLE
                # randomly choose 3 pixels, remove from list of indices
                init_edge_indices = np.random.choice(edge_indices, size=3, replace=False)

                init_edges = edges[init_edge_indices, :]
                # remove the edges from list of indices so we don't use them again 

                # Check to see if any of the pairwise distances between points
                #  are closer to each other than a threshold.
                # Here we do this using matrices which is more efficient
                dist_matrix = np.sum((init_edges[:, np.newaxis, :] - init_edges[np.newaxis, :, :]) ** 2, axis = -1)
                # look at only entires above upper diagonal
                if (dist_matrix[np.triu_indices(3, k=1)] < threshold_close).any():
                        print('initial points too close')
                        continue

                # Check if the points are co-linear, if so skip!
                # Recall: cross product of two vectors is 0 if they are colinear
                vec0 = init_edges[1]-init_edges[0]
                vec1 = init_edges[2]-init_edges[1]
                if np.cross(vec0, vec1) == 0:
                        print('co-linear points')
                        continue
                
                # Estimate our model parameters from the initial points
                # lots of ways of doing this, solve quadratic system of equations

                # Simpler method: Each of our points is of equal distance
                # from the centre of the circle (this is the radius)
                # Use this to create equations:
                # (x_1-a)^2 + (y_1-b)^2 = (x_2-a)^2 + (y_2-b)^2
                # 
                # working this out, quadratic terms cancel, gives us two linear equations
                # x_1^2 + 2*x_1*a + y_1^2 + 2*y_1*b = x_2^2 + 2*x_2*a + y_2^2 2*y_2*b 
                # ....
                # these are the solutions
                
                x1, y1 = init_edges[0]
                x2, y2 = init_edges[1]
                x3, y3 = init_edges[2]

                a = (x1*x1+y1*y1)*(y2-y3) + (x2*x2+y2*y2)*(y3-y1) + (x3*x3+y3*y3)*(y1-y2)
                a /= 2*(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2)
                
                b = (x1*x1 + y1*y1)*(x3-x2) + (x2*x2+y2*y2)*(x1-x3) + (x3*x3 + y3*y3)*(x2-x1)
                b /= 2*(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2)

                r = sqrt( (x1-a)**2 + (y1-b)**2 )

                # CONSENSUS
                # find edges close to our circle
                d = np.abs(np.sqrt((edges[:, 0] - a)**2 + (edges[:, 1] - b)**2) - r)
                inlier_count = np.sum(d <= threshold_inlier) + 3
                
                # check if we have enough inliers to consider this a good circle
                if inlier_count > 2*np.pi*r*inlier_threshold:
                        vote = inlier_count/(2*np.pi*r)
                        # we have a valid circle!
                        # save the parameters, along with the inlier count
                        candidates.append(circle(a, b, r, vote))

                        print("Circle: a={}, b={}, r={}, vote={}".format(a, b, r, vote))
                
        return candidates

def main():
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('filename', help='single channel 8-bit greyscale image filename')
        parser.add_argument('--thresholdlow', default=120, type=int, help='Canny low threshold')
        parser.add_argument('--thresholdhigh', default=180, type=int, help='Canny high threshold')
        parser.add_argument('--maxiterations', default=1000, type=int, help='Maximum number of iterations')
        parser.add_argument('--inlierdistance', default=4, type=int, help='distance within a point is considered an inlier')
        parser.add_argument('--mindistance', default=50, type=int, help='Minimum distance between initial points')
        parser.add_argument('--inlierthreshold', default=0.9, type=float, help='Inlier threshold')

        
        args = parser.parse_args()
        
        assert args.thresholdlow >= 0 and args.thresholdhigh >= args.thresholdlow, "threshold high >= threshold low"

        # read in the image from filename provided as first argument, convert to float32 representation
        image = cv2.imread(args.filename, cv2.IMREAD_GRAYSCALE)
        assert len(image.shape) == 2, "Error: input must be single-channel greyscale image!"
        
        cannyedges = find_edges(image, args.thresholdhigh, args.thresholdlow)
        #show_image(cannyedges, "Canny Edges")
        #write_image(cannyedges, "cannyedges.png")

        # find the indices of the non-zero pixels
        non_zero_row, non_zero_col = np.nonzero(cannyedges)
        non_zero = np.array(list(zip(non_zero_row, non_zero_col)))
        print('Found {} edge pixels out of {} total pixels'.format(len(non_zero), image.shape[0]*image.shape[1]))
        # find candidate circles with RANSAC
        circles = ransac_circle(non_zero, threshold_close=args.mindistance, threshold_inlier=args.inlierdistance, inlier_threshold=args.inlierthreshold, max_iterations=args.maxiterations)

        # display candidate circles
        maxvote = 0
        bestcircle = None
        for circle in circles:
                if circle.votes > maxvote:
                        bestcircle = circle
                        maxvote = circle.votes
        
        circleimage = cv2.cvtColor(cannyedges, cv2.COLOR_GRAY2BGR)
        for circle in circles:
                draw_circle(circleimage, circle, maxvote)
        
        show_image(circleimage, "Circles")

        bestcircleimage = cv2.cvtColor(cannyedges, cv2.COLOR_GRAY2BGR)
        draw_circle(bestcircleimage, bestcircle, maxvote)
        show_image(bestcircleimage, "Best Circle")


if __name__ == '__main__':
        main()