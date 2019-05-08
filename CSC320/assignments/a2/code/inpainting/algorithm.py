## CSC320 Winter 2019
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

## YOU DO NOT NEED TO UNDERSTAND THE CODE IN THIS FILE IN DETAIL. 
## THE ONLY MODIFICATIONS REQUIRED INVOLVE COPYING YOUR CODE FROM 
## A1 TO ENABLE IMAGE READING/WRITING AND INPUT VALIDATION

# import basic packages
import numpy as np
import cv2 as cv
import Queue
from patchdb import *
from psi import *
from copyutils import *
from debug import *

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you wish to import any additional modules
# or define other utility functions, 
# include them here

            
#########################################


#########################################
#
# The Inpainting Class
#
# This class contains the basic methods required for implementing 
# exemplar-based image inpainting. Description of
# the individual methods is given below.
#
# To run image inpainting one must create an instance
# of this class. See function run() in file run.py for an
# example of how it is called
#

class Inpainting:
    #
    # The class constructor
    #
    # When called, it creates a private dictionary object that acts 
    # as a container for all input and all output images of 
    # the inpainting algorithm. These images are initialized to None 
    # and populated/accessed by calling the the readImage(), writeImage(), 
    # and exampleBasedInpainting() methods.
    #
    def __init__(self):
        self._images = {
            'source':None,
            'alpha':None,
            'inpainted':None,
            'sourceGray':None,
            'fillFront':None,
            'confidence':None,
            'filled':None,
        }
        self._changedInput = True
        self._changedW = True
        self._w = None
        self._wFromGUI = 5
        self._maxIterations = 100
        self.initializeInpainting()
        self.debug = debug(patch=True, 
                            vectors=True, 
                            intensities=True,
                            verbose=True)
                
    # Use OpenCV to read an image from a file and copy its contents to the 
    # matting instance's private dictionary object. The key 
    # specifies the image variable and should be one of the
    # strings in lines 89-95. 
    #
    # The routine should return True if it succeeded. If it did not, it should
    # leave the matting instance's dictionary entry unaffected and return
    # False, along with an error message
    def readImage(self, fileName, key):
        success = False
        msg = 'No Image Available'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        retv = cv.imread(fileName)
        if retv is not None:
            if key == 'alpha' and retv.shape[2] == 3:
                self._images[key] = retv[:,:,0]
            else:
                self._images[key] = retv

            success = True
            msg = '%s read!'%fileName

        #########################################
        return success, msg

    # Use OpenCV to write to a file an image that is contained in the 
    # instance's private dictionary. The key specifies the which image
    # should be written and should be one of the strings in lines 89-95. 
    #
    # The routine should return True if it succeeded. If it did not, it should
    # return False, along with an error message
    def writeImage(self, fileName, key):
        success = False
        msg = 'No Image Available'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################

        retv = cv.imwrite(fileName, self._images[key])
        if retv:
            success = True
            msg = '%s written!'%fileName

        #########################################
        return success, msg


    #
    # Top-level method implementing the image inpainting algorithm. The
    # method takes its inputs/outputs from the method's private dictionary 
    # ojbect. 
    #
    # If maxIterations is specified, at most maxIterations iterations will
    # be executed before returning control to the GUI
    #
    #
    # IN THE FOLLOWING CODE, ALL REFERENCES TO STEPS 1,2, etc REFER TO
    # THE ALGORITHM IN TABLE 1 OF CRIMINISI ET AL, IEEE-TIP 2004.
    #
    def exampleBasedInpainting(self, imviewer, maxIterations=None):
        """
success, errorMessage = exampleBasedInpainting(self)
        
        Perform image inpainting. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
        """

        success = False
        msg = 'No Image Available'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################

        ## COPY/ADAPT THE CODE SNIPPETS FROM YOUR triangulationMatting()
        ## ROUTINE IN A1 THAT CHECK THE VALIDITY OF THE INPUT IMAGES
        ## Specifically: source and alpha must have the same dimesions,
        ## source much be a 3-channel uint8 image and alpha must be a 
        ## one-channel uint8 image.

        if self._images['source'] is not None and \
            self._images['alpha'] is not None:

            source = self._images['source']
            alpha = self._images['alpha']
            source_shape = self._images['source'].shape
            alpha_shape = self._images['alpha'].shape
                
            if  (source_shape[:2] == alpha_shape[:2]) and \
                (source_shape[2] == 3) and \
                (len(alpha_shape) == 2 or alpha_shape[2] == 1) and \
                (self._images['source'].dtype == np.uint8) and \
                (self._images['alpha'].dtype == np.uint8):

                success = True
                msg = 'Valid Image for Exemplar-based Inpainting'
        else:
            return success, msg

        #########################################

        #
        # Handle variable/data structure initialization
        #
        
        self.setPatchRadius(self.patchRadius())
        self.debug.setImviewer(imviewer)

        if (self.changedInput() or self.changedPatchRadius()):

            # 
            # The input has changed so we re-initialize all images used by
            # the algorithm and start the inpainting process from scratch
            
            self._images['filled'] = np.uint8(self._images['alpha'] > 0)*255
            self._images['inpainted'] = self._images['source'].copy()
            for i in range(0,3):
                self._images['inpainted'][:,:,i] *= (self._images['filled']>0)
        
            #
            # Step 1a,b: Identify the fill front deltaOmega and compute
            #            initial patch priorities
            
            self.computeBoundaries()
            self.confidenceInitialize()
            self.patchDBInitialize()
        
        
        self.iterationsInit()
        done = False

        #
        # This is the main loop
        #
        while ((not self.iterationsCompleted(maxIterations=maxIterations)) 
               and not done):
            done = self.inpaintRegion(imviewer=imviewer, 
                                      maxIterations=maxIterations)

        # Set a flag so that we don't have to re-execute the
        # algorithm if the run button is pressed again UNLESS the
        # input images have been changed through the GUI
        self.clearChangedInput()
        
        self.debug.initDisplay()
        
        return success, msg


    #
    # Method that implements Steps 1-3 of the paper
    #
    
    def inpaintRegion(self, imviewer=None, maxIterations=None):
        
        # If the input mask contains more than one masked region
        # we iterate over each region separately
        if self._fillNewRegion:
            try:
                # Use OpenCV to get the boundary curve of new masked region
                boundary = self._boundaryIterator.next()
                self.fillFrontInitialize(boundary, imviewer=imviewer)
                self._fillNewRegion = False
            except StopIteration:
                # Stop if no more masked regions can be found
                return True
        
        # Loop until the maximum number of iterations has been reached
        # or all image pixels have been inpainted
        while ((not self.fillFrontIsEmpty()) and 
               (not self.iterationsCompleted(maxIterations=maxIterations))):
               
            self.debug.clearDisplay()
               
            #
            # Step 2a: find the patch psiHatp with the highest priority 
            #          and remove it from the priority list
            #
            self._psiHatP = self.fillFrontGetHighestPriorityPatch()

            print("(C,D)={}".format((self._psiHatP._C, self._psiHatP._D)))
            
            if self.debug.verbose():
                print 'current patch: (%d,%d) [priority=%g]'%(
                        self._psiHatP.row(),
                        self._psiHatP.col(),
                        self._psiHatP.P()
                )

            self.debug.drawPatch(self._psiHatP, vectors=True, 
                                 red=0, green=0, blue=1)

            #
            # Step 2b: find the example patch, psiHatQ, in the 
            #          original image that is most similar to this 
            #          patch
            #
            bestRow, bestCol, rmsError, filledPixels, channels = \
                    self._patchDB.match(self._psiHatP.pixels(), 
                                        filled=self._psiHatP.filled(),
                                        returnValue=True)  
            self._psiHatQ = PSI((bestRow, bestCol), self._psiHatP.radius(),
                                image=self._images['inpainted'], 
                                filled=self._images['filled'])
            if self.debug.verbose():
                print 'best match: (%d,%d) with RMS error of %g over %d pixels in %d channels'%(
                        self._psiHatQ.row(),
                        self._psiHatQ.col(),
                        rmsError, 
                        filledPixels/channels, 
                        channels
                )

            self.debug.drawPatch(self._psiHatQ, red=0, green=1, blue=0)
            self.debug.printPatch(self._psiHatP, showFilled=True,
                                   text='psiHatP pixels before inpainting')
            self.debug.printPatch(self._psiHatQ, showFilled=True,
                                   text='psiHatQ pixels before inpainting')
                             

            #
            # Step 2c: copy into the unfilled pixels of psiHatP the
            #          corresponding pixels from psiHatQ and return
            #          a binary mask that indicates which pixels were
            #          newly filled
            #

            self._newlyFilled = self._psiHatP.canBeCopied(self._psiHatQ)

            copyutils.setWindow(self._images['inpainted'],
                                self._psiHatP._coords,
                                self._w,
                                self._psiHatQ.pixels(), 
                                condition=self._newlyFilled)

            if self.debug.showIntensities():
                print 'pixels to be inpainted:'
                print self._newlyFilled
            self.debug.printPatch(self._psiHatP, 
                                  text='psiHatP pixels after inpainting')

            ##
            ## Step 3: update the confidences of the newly-filled pixels
            ##            

            conf = PSI(self._psiHatP._coords, self._psiHatP.radius(),
                          image=self._images['confidence'], 
                          filled=self._images['filled'])

            self.debug.printPatch(conf, text='current confidences')

            self.confidenceUpdate(self._psiHatP, self._newlyFilled)

            self.debug.printPatch(conf, text='updated confidences')


            ##
            ## Step 1a: Update the set of unfilled pixels
            ##

            fill = PSI(self._psiHatP._coords, self._psiHatP.radius(),
                          image=self._images['filled'], 
                          filled=self._images['filled'])

            self.debug.printPatch(fill, text='current filled')

            copyutils.setWindow(self._images['filled'],
                      self._psiHatP._coords,
                      self._psiHatP._w,
                      255*np.ones_like(self._psiHatP.filled()))

            self.debug.printPatch(fill, text='updated filled')


            ##
            ## Step 1a: Update the fill front 
            ##

            front = PSI(self._psiHatP._coords, self._psiHatP.radius()+2,
                          image=self._images['fillFront'], 
                          filled=self._images['filled'])

            self.debug.printPatch(front, text='current front')

            self.fillFrontUpdate()

            self.debug.printPatch(front, text='updated front')
            
            #
            # Step 1b: (re)compute the confidence & priority of each patch 
            #          on the fill front (ie. the priority list)
            #
            self.recomputePatchPriorities()

            self.iterationsNew()
            
        if self.fillFrontIsEmpty():
            self._fillNewRegion = True
            
        return self.iterationsCompleted(maxIterations=maxIterations)
        
        
    #
    # Helper methods for implementing individual steps of the algorithm
    #        
        

    # build the patch database
    def patchDBInitialize(self):
        self._patchDB = PatchDB(self._images['inpainted'], self._w, filled=self._images['filled'])

    # use the OpenCV contour-finding routine to compute the boundary
    # of a masked region in the image
    def computeBoundaries(self):
        # The OpenCV contour-finding routine returns a boundary whose pixels
        # are on the non-zero side of the boundary. To compute a
        # boundary whose pixels are on the zero side, we invert the 
        # mask and apply the contour-finder on that image
        unfilled = np.uint8(self._images['filled'] == 0)
        _, boundaries, _ = cv.findContours(unfilled, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        self._boundaryIterator = iter(boundaries)

    # check if all pixels have been inpainted
    def fillFrontIsEmpty(self):
        return self._deltaOmega.empty()
        
    # initialize the image containing the fill front. we do this
    # by calling a curve-drawing OpenCV function that takes as 
    # input the ordered list of pixels on the boundary of a masked region
    #
    # we then create a python priority queue into which we push all
    # patches corresponding to boundary pixels
    def fillFrontInitialize(self, boundaryPixels, imviewer=None):
        # initialize the pixels on the fill front
        self._images['fillFront'] = \
            np.zeros_like(self._images['filled'], dtype=np.uint8)
        self._images['fillFront'] = \
            cv.drawContours(self._images['fillFront'],boundaryPixels,-1,255)
        
        # initialize the priority queue with all points on the fill front
        self._deltaOmega = Queue.PriorityQueue()
        for colrow in boundaryPixels:
            col, row = colrow[0]
            
            # create a new patch object centered at the pixel
            p = PSI((row,col), self._w, 
                      image=self._images['inpainted'], 
                      filled=self._images['filled'],
                      confidence=self._images['confidence'],
                      fillFront=self._images['fillFront'])
            self._deltaOmega.put(p)
            
    # get the lowest-priority patch from the queue
    # since our priorities are always negative, this returns the patch
    # with the highest P() value according to the algorithm in the paper
    def fillFrontGetHighestPriorityPatch(self):
        return self._deltaOmega.get()
    
    # once a patch on the fill front has been inpainted, we need to
    # update the fill front (both the fill front image and the priority queue)
    def fillFrontUpdate(self):
        # clear the fillFront flag for all pixels inside psiHatP
        copyutils.setWindow(self._images['fillFront'],
                  self._psiHatP._coords,
                  self._psiHatP._w,
                  np.zeros_like(self._psiHatP.filled()))
        # get the coordinates of all pixels that are immediately
        # outside the patch and within the image limits
        borderCoords = copyutils.outerBorderCoords(self._images['inpainted'], 
                                         self._psiHatP._coords,
                                         self._w)
        # make sure those pixels are not filled and not on the fill front 
        # already
        addToFillFront = lambda x: (self._images['fillFront'][x[0],x[1]]==0 and
                                    self._images['filled'][x[0],x[1]]==0)
        newFillFrontCoords = filter(addToFillFront, borderCoords)

        # now add all those coordinates to the priority list and set their
        # fillFront flag 
        for rowcol in newFillFrontCoords:
            row, col = rowcol
            self._images['fillFront'][row, col] = 255
            p = PSI((row, col), self._w, 
                      image=self._images['inpainted'], 
                      filled=self._images['filled'],
                      confidence=self._images['confidence'],
                      fillFront=self._images['fillFront'])
            self._deltaOmega.put(p)

    # initialize the image where patch confidences are stored
    def confidenceInitialize(self):
        self._images['confidence'] = 255.0*(self._images['filled'] > 0)
        
    # once the confidence of a patch on the fill front has been computed,
    # we assign that confidence to all unfilled pixels in the patch
    def confidenceUpdate(self, p, newlyFilled):
        # copy the updated confidences back to the confidence image
        # but only for the newly-filled pixels
        conf, valid = copyutils.getWindow(self._images['confidence'], 
                                          p._coords, p._w)
        copyutils.setWindow(self._images['confidence'], p._coords, p._w, 
                            newlyFilled*p._C, condition=newlyFilled)

    # re-compute the priorities of all patches. we need to do this
    # each time after modifying the inpainted image 
    def recomputePatchPriorities(self):
        deltaOmega2 = Queue.PriorityQueue()
        # remove all patches from the original queue, recompute
        # their priorities, and put back in a new queue
        while (not self._deltaOmega.empty()):
            try:
                psiHatP = self._deltaOmega.get()
                
                # if that patch center has already been filled, we don't
                # update its priority and don't put it back on the priority list
                row, col = psiHatP._coords
                if self._images['filled'][row, col]:
                    pass
                else:
                    psiHatP.updateC(confidence=self._images['confidence'],
                                    filled=self._images['filled'])
                    psiHatP.updateD(filled=self._images['filled'],
                                    fillFront=self._images['fillFront'],
                                    inpainted=self._images['inpainted'])
                    psiHatP.updateP()
                    deltaOmega2.put(psiHatP)
            except Queue.Empty:
                break
        # update the original queue 
        self._deltaOmega = deltaOmega2

#
# Miscellaneous helper functions
#

    def initializeInpainting(self):
        self._fillNewRegion = True
        
    def iterationsInit(self):
        self._iterationsDone = 0
        
    def iterationsCompleted(self, maxIterations=None):
        if maxIterations is None:
            maxit = self.maxIterations()
        else:
            maxit = maxIterations
        return (maxit != -1) and (self._iterationsDone >= maxit)
        
    def iterationsNew(self):
        if self.debug.verbose():
            print 'Finished iteration', self._iterationsDone, '\n'
        self._iterationsDone += 1
        
    def maxIterations(self):
        return self._maxIterations
        
    def setMaxIterations(self, value):
        self._maxIterations = value

    def patchRadius(self):
        return self._wFromGUI
        
    def setPatchRadius(self, value):
        self._wFromGUI = value
        if self._w != value:
            self._w = value
            self._changedW = True
            self.initializeInpainting()
            self.setChangedInput()
        else:
            self._changedW = False

    def setChangedInput(self):        
        self.initializeInpainting()
        self._changedInput = True

    def clearChangedInput(self):
        self._changedInput = False

    def changedInput(self):
        return self._changedInput
        
    def changedPatchRadius(self):
        return self._changedW

