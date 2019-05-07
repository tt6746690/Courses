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

# import basic packages
import numpy as np
from psi import *

from kivy.properties import ObjectProperty
from kivy.graphics import *


class debug:
    def __init__(self, patch=False, vectors=False, 
                 intensities=False, verbose=False,
                 imviewer=None):
        self._showPatch = patch
        self._showVectors = vectors
        self._showIntensities = intensities
        self._verbose = verbose
        self._imviewer = imviewer

    def setImviewer(self, imviewer):
        self._imviewer = imviewer
        
    def setShowPatch(self, value):
        self._showPatch = value        
        
    def setShowVectors(self, value):
        self._showVectors = value

    def setShowIntensities(self, value):
        self._showIntensities = value
    
    def setVerbose(self, value):
        self._verbose = value
    
    def verbose(self):
        return self._verbose
    
    def showPatch(self):
        return self._showPatch
    
    def showVectors(self):
        return self._showVectors
    
    def showIntensities(self):
        return self._showIntensities

    def printPatch(self, psi, channel=None, showFilled=False, text=''):
        if self.showIntensities():
            if showFilled:
                print text, '(filled):'
                psi.printFilled()
            if channel is not None:
                print text,' (intensities):'
                psi.printChannel(channel)
            else:
                for i in range(0,psi.numChannels()):
                    print text,' (channel %d)'%i
                    psi.printChannel(i)
                    
    def clearDisplay(self):
        if self._imviewer is not None:
            self._imviewer.draw_remove_group('show_patch')
            self._imviewer.draw_remove_group('show_vectors')
            
    def initDisplay(self):
        if self._imviewer is not None:
            if self.showPatch():
                self._imviewer.draw_enable_group('show_patch')
            else:
                self._imviewer.draw_disable_group('show_patch')
            if self.showVectors():
                self._imviewer.draw_enable_group('show_vectors')
            else:
                self._imviewer.draw_disable_group('show_vectors')
            self._imviewer.draw_enabled()
        
                    
    def drawPatch(self, psi, vectors=False, 
                  red=1, green=1, blue=1, 
                  gred=1, ggreen=0, gblue=0,
                  nred=0, ngreen=1, nblue=0):
        if self._imviewer is not None:
            if self.showPatch():
                self._imviewer.draw_color(red=red, 
                                          green=green, 
                                          blue=blue, 
                                          group='show_patch')
                self._imviewer.draw_rectangle_centered(
                        r=psi.row(), 
                        c=psi.col(), 
                        radius=psi.radius(),
                        group='show_patch')
            if vectors and self.showVectors():
                print 'entered vector drawing'
                gcoords = psi.grad()
                coords = (psi.row(), psi.col())
                gmag = np.sqrt(gcoords[0]*gcoords[0]+gcoords[1]*gcoords[1])
                gr, gc = gcoords[0]/gmag, gcoords[1]/gmag
                angle = np.arctan2(gc, gr)
                self._imviewer.draw_color(red=gred, green=ggreen, blue=gblue,
                                          group='show_vectors')
                self._imviewer.draw_vector(
                        r=psi.row(), 
                        c=psi.col(), 
                        angle=angle,
                        arrow=0.3,
                        length=5*psi.radius(), 
                        group='show_vectors')
                nr, nc = psi.normal()
                if (nr is not None) and (nc is not None):
                    angle = np.arctan2(nc, nr)
                    self._imviewer.draw_color(red=nred, green=ngreen, 
                                              blue=nblue, group='show_vectors')
                    self._imviewer.draw_vector(
                            r=psi.row(), 
                            c=psi.col(), 
                            angle=angle,
                            arrow=0.3,
                            length=5*psi.radius(), 
                            group='show_vectors')
                