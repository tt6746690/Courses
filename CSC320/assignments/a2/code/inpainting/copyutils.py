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

#
# This file implements the core patch-copying routines. These
# routines are used in numerous places in the code to (1) extract
# pieces of an image that correspond to a patch and (2) copy
# patch pixels to a new image location. 
#

#
# getWindow: retrieve a (2w+1)x(2w+1) block of pixels from an OpenCV
#            image and place it into a numpy array
#
# Input arguments:
#    image:            the source image
#    coords:           a sequence of the form (row,col) indicating
#                      the patch center
#    w:                the radius of the patch
#    outofboundsvalue: if the patch extends beyond the image boundary
#                      this value will be used for the intensity of
#                      the out-of-bounds pixels
# Return values:
#    - a numpy array of size (2w+1)x(2w+1)xC where C is the number of 
#      color channels in the source image. If the source image has only
#      one color channel, a 2D matrix of size (2w+1)x(2w+1) is returned
#    - a binary matrix of size (2w+1)x(2w+1) whose pixels are True if and
#      only if they lie inside the image boundary
#
def getWindow(image, coords, w, outofboundsvalue=0):
    row, col = coords
    if len(image.shape) == 2:
        image0 = image[:,:,None]
    else:
        image0 = image
    partial = False
    if row-w < 0:
        firstDestRow = w-row
        numRows = (2*w+1)-firstDestRow
        firstSrcRow = 0
        partial = True
    else:
        firstDestRow = 0
        numRows = 2*w+1
        firstSrcRow = row-w
    if col-w < 0:
        firstDestCol = w-col
        numCols = (2*w+1)-firstDestCol
        firstSrcCol = 0
        partial = True
    else:
        firstDestCol = 0
        numCols = 2*w+1
        firstSrcCol = col-w
    if row+w >= image0.shape[0]:
        numRows -= ((row+w+1)-image0.shape[0])
        partial = True
    if col+w >= image0.shape[1]:
        numCols -= ((col+w+1)-image0.shape[1])
        partial = True

    window = np.full((2*w+1,2*w+1,image0.shape[2]),
                     outofboundsvalue,dtype=image0.dtype)
    window[firstDestRow:(firstDestRow+numRows),
           firstDestCol:(firstDestCol+numCols), :] = \
        image0[firstSrcRow:(firstSrcRow+numRows),
               firstSrcCol:(firstSrcCol+numCols), :]
    if not partial:
        valid = np.full((2*w+1,2*w+1), True, dtype=np.uint8)
    else:
        valid = np.zeros((2*w+1,2*w+1), dtype=np.uint8)
        valid[firstDestRow:(firstDestRow+numRows),
              firstDestCol:(firstDestCol+numCols)] = True

    return np.squeeze(window), valid

#
# setWindow: copies a (2w+1)x(2w+1) array of pixels to the specified location
#            in an image
#
# Input arguments:
#    image:            the destination image
#    coords:           a sequence of the form (row,col) indicating
#                      the patch center
#    w:                the radius of the patch
#    pixels:           a numpy matrix of size (2w+1)x(2w+1)xC containing the
#                      patch pixels
#    condition:        a binary matrix of size (2w+1)x(2w+1) that pixel copying:
#                      a pixel (r,c) will be copied from array pixels to the 
#                      destination image if and only if condition(r,c)=True
#                      If condition=None then all pixels are copied
#                      
def setWindow(image, coords, w, pixels, condition=None):
    row, col = coords
    if len(image.shape) == 2:
        image0 = image[:,:,None]
    else:
        image0 = image
    if len(pixels.shape) == 2:
        pixels0 = pixels[:,:,None]
    else:
        pixels0 = pixels

    if row-w < 0:
        firstSrcRow = w-row
        numRows = pixels0.shape[0]-firstSrcRow
        firstDestRow = 0
    else:
        firstSrcRow = 0
        numRows = pixels0.shape[0]
        firstDestRow = row-w
    if col-w < 0:
        firstSrcCol = w-col
        numCols = pixels0.shape[1]-firstSrcCol
        firstDestCol = 0
    else:
        firstSrcCol = 0
        numCols = pixels0.shape[1]
        firstDestCol = col-w
    if row+w >= image0.shape[0]:
        numRows -= ((row+w+1)-image0.shape[0])
    if col+w >= image0.shape[1]:
        numCols -= ((col+w+1)-image0.shape[1])

    if condition is not None:
        for c in range(0,image0.shape[2]):
            srcPixels = np.squeeze(image0[firstDestRow:(firstDestRow+numRows), 
                                          firstDestCol:(firstDestCol+numCols), c])
            destPixels = np.squeeze(pixels0[firstSrcRow:(firstSrcRow+numRows),
                                           firstSrcCol:(firstSrcCol+numCols), c])
            condPixels = condition[firstSrcRow:(firstSrcRow+numRows),
                                   firstSrcCol:(firstSrcCol+numCols)]           
            srcPixels[condPixels>0] = destPixels[condPixels>0]

            image0[firstDestRow:(firstDestRow+numRows), 
                   firstDestCol:(firstDestCol+numCols), c] = srcPixels
    else:
        image0[firstDestRow:(firstDestRow+numRows), 
               firstDestCol:(firstDestCol+numCols), :] = \
            pixels0[firstSrcRow:(firstSrcRow+numRows),
                    firstSrcCol:(firstSrcCol+numCols), :]


#
# Given an image, a patch center and a patch radius, return an ordered list of 
# 2D coordinates corresponding to the pixels that (a) lie just outside the patch
# and (b) are contained within the image boundary
#
def outerBorderCoords(image, coords, w):
    wo = w+1
    row, col = coords
    rows = np.arange(row-wo,row+wo+1)
    rowplus = np.full_like(rows,row+wo)
    rowminus = np.full_like(rows,row-wo)
    # we don't want the same coords appearing twice so the
    # horizontal outer boundary contains one less pixel
    cols = np.arange(col-wo,col+wo)  
    colplus = np.full_like(cols,col+wo)
    colminus = np.full_like(cols,col-wo)
    borderCoords =  (zip(rows,colplus) + 
                     zip(rowplus,cols) + 
                     zip(rows,colminus) + 
                     zip(rowminus,cols))
    withinLimits = lambda x: ((x[0]<image.shape[0]) and 
                            (x[0]>=0) and
                            (x[1]<image.shape[1]) and
                            (x[1]>=0))
    return filter(withinLimits, borderCoords)
    