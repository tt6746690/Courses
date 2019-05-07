## CSC320 Winter 2019
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE 
##

import numpy as np
import copyutils
import cv2 as cv
from compute import computeC, computeGradient, computeNormal

#########################################
#
# The Class PSI
#
# This class contains all methods required for dealing with 
# image patches. Description of the individual methods is 
# given below.
#
    
class PSI:
    #
    # The class constructor
    # 
    # At a minimum, the constructor takes as input two arguments:
    #    coords: a 2D numpy vector containing the (row,col) coordinates
    #            of the patch center
    #    w:      the radius of the patch (ie. the size of the patch is 
    #            (2w+1)x(2w+1) )
    # The constructor also takes additional arguments that specify the
    # OpenCV images containing the inpainted image I, the binary image 
    # indicating which pixels in I are filled, the binary image containing
    # the fill front, and the image that stores the confidence of the 
    # already-inpainted pixels.
    # 
    # The constructor is responsible for computing the initial priority
    # of the patch, by calling the gradient, normal, and confidence 
    # computation routines in compute.py
    #
    # The constructor also takes care of cases where the patch goes beyond
    # the image border. The patch data structure includes a binary matrix 
    # of size (2w+1)x(2w+1) that indicates which pixels in the patch are valid,
    # ie. are within the image border.
    # 
    def __init__(self,              
                 coords,            # 2D numpy array containing vector (row,col)
                 w,                 # the patch radius
                 image=None,        # the color image I being inpainted
                 filled=None,       # the uint8 image indicating filled pixels
                 fillFront=None,    # the uint8 image storing the fill front
                 confidence=None):  # the uint8 image storing pixel confidences 
        assert len(coords) == 2

        # coordinates of patch center
        self._coords = coords 
        # patch radius
        self._w = w
        # the constant used in equation defining D(p) on p.4 of paper
        self._alpha = 255           
        # patch priority P(p)
        self._P = None
        # patch confidence C(p)
        self._C = None
        # patch data term D(p)
        self._D = None
        # pointer to the OpenCV image being inpainted
        self._img = image
        # pointer to the OpenCV image containing the filled pixels
        self._fld = filled
        
        # Matrix that transforms OpenCV (row,col) coordinates to 
        # patch (row,col) coordinates, which are relative to the patch
        # center, take values in the range [-w,w], and have point (-w,-w)
        # at the lower-left corner of the patch. 
        self._im2mat = np.array([[1, 0, -coords[0]+w+1],
                                 [0, 1, -coords[1]+w+1],
                                 [0, 0, 1]])
        # 3x3 matrix for performing the inverse transformation.
        self._mat2im = np.linalg.inv(self._im2mat)

        # compute the confidence of the patch center using the equation
        # for C(p) on p.4 of the paper
        self.updateC(confidence=confidence, filled=filled)
        # compute the data term of the patch center using the equation
        # for D(p) on p.4 of the paper
        self.updateD(filled=filled, fillFront=fillFront, inpainted=image)
        # compute the priority of the patch using Eq.(1) in the paper
        self.updateP()

    #
    # Core methods of the class
    #
                                
    #
    # 
    # Method that returns a numpy array containing the pixels
    # of the patch. You are likely to use this function repeatedly
    # in your implementation
    #
    # Note that the pixels are *not* explicitly stored
    # in the patch. Rather, they are copied directly from the image
    # whenever this method is called. This is because the images change
    # during the course of the algorithm and we need to make sure that
    # the pixels retrieved are always up to date.
    #
    # The method takes care of cases where the patch extends beyond the
    # image border. In that case, the out-of-bounds pixels have the value
    # zero. If the input argument returnValid is set to True, the method
    # returns a second binary array that is zero at all invalid patch pixels
    # ie., pixels that extend beyond the image border
    #
    def pixels(self, returnValid=None):
        if self._img is not None:
            # if the window extends beyond the image limits, we need
            # to store a bitmap that indicates which pixels in the 
            # patch are actually valid
            pix, valid = copyutils.getWindow(self._img, self._coords, self._w)
        else:
            pix = None
        if len(pix.shape) == 2:
            pix = pix[:,:,None]
        if returnValid is None:
            return pix
        else:
            return pix, valid

    #
    # This method is identical to the one above, except that it returns
    # a binary array indicating which pixels in the patch have already
    # been filled (or that were not masked in the original image)        
    #
    def filled(self):
        if self._fld is not None:
            # if a window extends beyond image limits, the fill value for the
            # out-of-bounds pixels is set to zero (ie. those patch pixels are not "filled")
            fill, _ =  copyutils.getWindow(self._fld, self._coords,
                                           self._w, outofboundsvalue=False)
        else:
            fill = np.fill((2*w+1,2*w+1),True,dtype=np.uint8)
        return fill        
        
    #
    # This method provides the basic mechanism for copying
    # pixels between two patches
    #
    # given two patches, self and other, the method  
    # returns a binary matrix of size (2w+1)x(2w+1)
    # which contains True for every pixel (r,c) in self that (a) is
    # not yet filled and (b) the corresponding pixel (r,c) in other
    # is filled and valid
    #
    def canBeCopied(self, other):
        pix = self.pixels()
        oth, valid = other.pixels(returnValid=True)
        assert len(pix.shape) == 3
        assert len(oth.shape) == 3
        
        # copy those pixels from the other patch that (a) are valid
        # and (b) correspond to unfilled pixels in the current patch
        copied = np.logical_and(self.filled() == 0, valid > 0)
        return copied
        
    #
    # Compute the confidence C(p) of the patch center by calling the
    # corresponding function in compute.py
    #
    def updateC(self, confidence=None, filled=None):
        if (confidence is not None) and (filled is not None):
            self._C = computeC(psiHatP = self, 
                               confidenceImage = confidence, 
                               filledImage = filled)

    #
    # Compute the data term D(p) of the patch center by calling the
    # corresponding function in compute.py
    #
    def updateD(self, filled=None, fillFront=None, inpainted=None):
        if (filled is not None) and (fillFront is not None) and (inpainted is not None):
            nr, nc = computeNormal(psiHatP=self, 
                                   fillFront=fillFront, 
                                   filledImage=filled)
            self._normal = (nr, nc)
            gr, gc = computeGradient(psiHatP=self, 
                                     inpaintedImage=inpainted, 
                                     filledImage=filled)
            self._grad = (gr, gc)
            if (nr is not None) and (nc is not None):
                self._D = np.abs(nr*gc-nc*gr)/self._alpha
            else:
                self._D = 1
            
    #
    # Compute the patch priority P(p) of the patch center by calling the
    # corresponding function in compute.py
    #
    # Note that the implementation computes the *negative* of the
    # quantity defined in the paper because the python priority queue 
    # returns the element with the lowest priority value
    #
    def updateP(self):
        if (self._C is not None) and (self._D is not None):
            self._P = -(1.0*self._C/255)*self._D
        elif (self._C is not None):
            self._P = -(1.0*self._C/255)
        elif (self._D is not None):
            self._P = -self._D

        
    #
    # Utility functions for accessing patch properties, etc
    #
        
    # we define a patch comparator so that patches can be placed in 
    # a priority list and sorted according to their priority
    def __cmp__(self, other):
        return cmp(self._P, other._P)

    # accessor for patch priority
    def P(self):
        return self._P
        
    # accessor for patch confidence
    def C(self):
        return self._C
        
    # accessor for the patch gradient
    def grad(self):
        return self._grad

    # accessor for the patch normal
    def normal(self):
        return self._normal
    
    # image row of the patch center
    def row(self):
        return self._coords[0]
        
    # image column of the patch center
    def col(self):
        return self._coords[1]
    
    # radius of the patch
    def radius(self):
        return self._w        

    # return the number of channels of the 
    # patche's pixels
    def numChannels(self):
        if self._img is not None:
            if len(self._img.shape) == 3:
                return self._img.shape[2]
            else:
                return 1
        else:
            return 0
        
    # print the numpy matrix containing the patche's pixel
    # intensities for a specific color channel 
    def printChannel(self, chan):
        print np.squeeze(self.pixels()[:,:,chan])
        
    # print the numby matrix indicating which patch pixels
    # have already been inpainted    
    def printFilled(self):
        print self.filled()
        
    # return the coordinates of OpenCV pixel (r,c) in the
    # coordinate system of the patch
    def im2mat(self, r, c):
        rc = np.dot(self._im2mat, np.array([[r],[c],[1]]))
        return rc[0], rc[1]
        
    # return the OpenCV coordinates of a pixel whose patch-centered
    # coordinates are (r,c)
    def mat2im(self, r, c):
        rc = np.dot(self._mat2im, np.array([[r],[c],[1]]))
        return rc[0], rc[1]
              
        
        
    