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
import cv2 as cv


#########################################
#
# The Class patchDB
#
# This class implements a patch database that allows retrieval
# of the patch in the database that is most similar to a query patch, 
# while also taking into account the fact that some of the query patche's
# pixels may be unknown
#
# You do NOT need to look at this class for your A2 implementation
#

class PatchDB:
    #
    # Database constructor
    #
    # Given an image, a patch radius w, and a binary image that indicates
    # which pixels are filled/known, the method creates a 2D matrix of 
    # of size (2w+1)*(2w+1) x N where N is the set of patches whose pixels
    # are all filled/known
    #
    def __init__(self, im, w, filled=None):
        if len(im.shape) == 2:
            # grayscale image
            im0 = im[:,:,None]
            channels = 1
        else:
            im0 = im
            channels = im.shape[2]
        #
        validRows = im0.shape[0]-2*w
        assert validRows > 0

        rowVec = np.arange(-w,w+1)[:,None]
        rowIndices = np.dot(rowVec,np.ones((1,rowVec.size),dtype=np.int32))
        rowIndicesVec = rowIndices.reshape((1,-1))
        #
        validCols = im0.shape[1]-2*w
        assert validCols > 0
        colVec = np.arange(-w,w+1)[None,:]
        colIndices = np.dot(np.ones((colVec.size,1),dtype=np.int32),colVec)
        colIndicesVec = colIndices.reshape((1,-1))
        #
        self._patches = np.zeros((rowIndices.size, validRows*validCols, channels), dtype=np.uint8)
        self._rindices = np.dot(np.arange(w,validRows+w)[:,None],np.ones((1,validCols), dtype=np.int32)).reshape((1,-1))
        self._cindices = np.dot(np.ones((validRows,1),dtype=np.int32),np.arange(w,validCols+w)[None,:]).reshape((1,-1))
        # erode the filled region to leave only pixels whose
        # windows do not intersect the unfilled region
        if filled is not None:
            kernel = np.ones((2*w+1,2*w+1),dtype=np.uint8)
            valid2D = cv.erode(filled, kernel, iterations=1)
        else:
            valid2D = np.full_like(im0[:,:,0],255)
        self._valid = valid2D[w:w+validRows, w:w+validCols].reshape((1,-1))
        #
        for i in range(0,colIndicesVec.size):
            for c in range(0,channels):
                rowOffset = rowIndicesVec[0,i]+w
                colOffset = colIndicesVec[0,i]+w
                self._patches[i,:,c] = im0[rowOffset:rowOffset+validRows, 
                                           colOffset:colOffset+validCols, 
                                           c].flatten()

    #
    # Patch matching method: given a query patch p and a binary array filled 
    # that indicates which pixels in p are known, it finds the most similar
    # patch stored in the database according to SSD similarity
    #
    def match(self, p, filled=None, returnValue=False):
        assert len(p.shape) == 3
        
        channels = p.shape[2]
        assert p.shape[0]*p.shape[1] == self._patches.shape[0]
        assert p.shape[2] == self._patches.shape[2]

        mshape = (p.shape[0],p.shape[1])
        if filled is None:
            filled = np.ones(mshape,dtype=np.uint8)
        assert filled.shape == mshape
        filledVec = filled.flatten()
        
        ssd = np.zeros((1,self._patches.shape[1]))
        # 
        large = p.size*255*255*(self._valid == 0)
        #
        filledPixels = 0
        for c in range(0,channels):
            pVec = p[:,:,c].flatten()
            for i in range(0,len(pVec)):
                if filledVec[i] > 0:
                    diff = np.squeeze(self._patches[i,:,c]) - 1.*pVec[i]
                    ssd += np.power(diff,2) + large
                    filledPixels += 1
                
        argmin = np.argmin(ssd)
        if returnValue:
            return self._rindices[0,argmin], self._cindices[0,argmin], np.sqrt(np.amin(ssd)/filledPixels), filledPixels, channels
        else:
            return self._rindices[0,argmin], self._cindices[0,argmin]
