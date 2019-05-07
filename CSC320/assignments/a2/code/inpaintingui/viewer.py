## CSC320 Winter 2019 
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT AT THE VERY END, 
## AS INDICATED
##

##
## This file defines the ImageViewer widget class. This 
## class manages all image dispay functionalities of the GUI.
## It is a generic image display widget that does not depend
## on the inpainting algorithm in any way.
##
## This class was adapted from Kivy's Pictures tutorial example
## and relies on Kivy's built-in Scatter widget class.
##

import kivy
kivy.require('1.9.1')

import io
import sys
import numpy as np
import cv2 as cv

from kivy.logger import Logger
from kivy.uix.scatter import Scatter
from kivy.uix.label import Label
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty
from kivy.properties import ObjectProperty
from kivy.graphics import *

#
# Convert from OpenCV's internal memory representation for
# images to Kivy's internal memory representation for images.
# There is likely an easier way to do this but I haven't
# quite found it. For the time being, I convert the image to
# from OpenCV to PNG and then from PNG to Kivy. These
# conversions are performed in memory (ie. nothing written to disk)
#
def openCV_to_kivy(cvImage):
    # encode the OpenCV image in PNG format
    _, imPNG = cv.imencode(".png", cvImage)
    # create a binary data stream for reading that data
    data = io.BytesIO(imPNG.tobytes())
    # create a Kivy Core Image data structure to hold that data
    return CoreImage(data, ext="png")

#
# A user-defined widget class for image display, derived from Kivy's
# built-in Scatter widget class
#
class ImageViewer(Scatter):
    source = StringProperty(None)
    centerInitialized = False
    # initial position of bottom-left corner of image frame when no image has been loaded yet
    init_pos = [106,82]
    display_list = dict([])
    display_list_groups = dict([])
    
    # Display an OpenCV image by creating a Kivy Texture object whose
    # texture memory stores the OpenCV image
    # I am not using mip-mapped Kivy textures for the time being but this
    # really should be done... 
    def display_opencv_image(self, im = None, name = None):
        print('display_opencv_image({})'.format(name))
        if im is not None:
            # convert an image in OpenCV's native in-memory format to
            # kivy's in-memory format
            kivyImage = openCV_to_kivy(im)
            # display the image over a white background
            self.ids.image.color = [1,1,1,1]
            # set the kivy Image's texture data 
            self.ids.image.texture = kivyImage.texture
            # make sure the image is displayed at its full size
            self.resize(kivyImage.size,0)
            # store the initial position of the image so we can
            # move the image back there if we wish to do so
            self.init_pos = self.pos
            # compute the 3x3 openCV to kivy coordinate transformation
            # that maps openCV (row,col) coordinates to kivy (x,y) coordinates
            self._openCVRowColToKivyXY = np.array([[0,  1,  0],
                                                  [-1, 0,  im.shape[0]-1],
                                                  [0,  0,  1]])
        else:
            # if we don't have an image, just display a black background
            # in its place
            self.ids.image.color = [0,0,0,1]

    # Set the size parameter of the Kivy Image object so that the image
    # it stores is drawn in its entirety on the canvas. Since the image
    # contains a border as well as a shadow region around it, we must
    # account for them when deciding on the total size of the image in pixels
    def resize(self, size, shadow):
        # reduce the image size by the amount needed for shadow rendering
        newSize = [size[0]-shadow*2, size[1]-shadow*2]
        normSize = self.ids.image.norm_image_size
        # udpate the image size while preserving aspect ratio
        aspect = self.ids.image.image_ratio
        if aspect > 1.0:
            if newSize[0]/aspect > newSize[1]:
                # image fits in the window at full width
                newSize2 = [newSize[0], newSize[0]/aspect]
            else:
                newSize2 =  [newSize[1]*aspect, newSize[1]]
        else:
            if newSize[0]/aspect > newSize[1]:
                # image fits in the window at full width
                newSize2 = [newSize[1]*aspect, newSize[1]]
            else:
                newSize2 =  [newSize[0], newSize[0]/aspect]
            
        self.ids.image.size = newSize2
            
    # Reposition the Kivy image
    def repos(self, pos, shadow):
        self.pos = [pos[0]+shadow, pos[1]+shadow]

    # function to handle a mouse button pressed event
    def on_touch_down_callback(self, touch):
        # we are only interested in mouse button events
        if 'button' not in touch.profile:
            return
        # we are only interested in left mouse button events
        if (touch.button != 'left'):
            return
        # if a left-double-tap is detected, we reset the image display
        if touch.is_double_tap:
            # if the image was zoomed in or out, reset its scale to 1
            self.scale=1
            # if the image was rotated, reset its rotation angle to zero
            self.rotation=0
            # move image to the position when it was originally constructed 
            # and displayed
            self.pos = self.init_pos
            # delete any axes we've already drawn
            if self.ud.has_key('group'):
                self.canvas.remove_group(self.ud['group'])
        else:    
            # get the user-defined data dictionary for the touch event
            self.ud = touch.ud
            # get the unique ID of the touch event and store it in the
            # dictionary
            self.ud['group'] = str(touch.uid)
            # store the list of OpenGL drawing commands for drawing the axes
            with self.canvas:
                # command to draw the axes in red
                Color(1, 0, 0, mode='rgb', group=self.ud['group'])
                # commands to draw the lines themselves
                self.ud['lines'] = [
                    Line(points=[touch.pos[0], 0,
                                 touch.pos[0], self.ids.image.height
                             ], group=self.ud['group']),
                    Line(points=[0, touch.pos[1],
                                 self.ids.image.width, touch.pos[1], 
                             ], group=self.ud['group'])
                    ]
            # create a label widget that will display the pixel position 
            self.ud['label'] = Label(size_hint=(None, None))
            # update the label's content
            self.update_touch_label(self.ud['label'], touch)
            # display the widget
            self.add_widget(self.ud['label'])


    def update_touch_label(self, label, touch):
        # create a new text string for the label
        label.text = '(x,y) = (%d, %d)' % (touch.x, touch.y)
        # trigger a refresh of the label's contents 
        label.texture_update()
        # reposition the label
        label.pos = touch.pos
        label.size = label.texture_size[0] + 20, label.texture_size[1] + 20     
                           
#
# Methods for converting OpenCV (r,c) coordinates to Kivy (x,y) coordinates
#

    def openCVRowColToKivyXY(self, r, c):
        pt = np.dot(self._openCVRowColToKivyXY, np.array([[r],[c],[1]]))
        return pt[0], pt[1]

    def openCVRowColToKivyXYPts(self, mat):
        return np.dot(self._openCVRowColToKivyXY, mat)


#
# Methods for drawing basic geometric shapes over images in Kivy
#

    def draw_set_group(self, group):
        if not self.display_list.has_key(group):
            self.display_list[group] = []
            self.display_list_groups[group] = False

    def draw_remove_group(self, group):
        if self.display_list_groups.has_key(group):
            del self.display_list_groups[group]
        if self.display_list.has_key(group):
            self.canvas.remove_group(group)
            del self.display_list[group]
            
    def draw_remove_all(self):
        for g in self.display_list_groups:
            self.canvas.remove_group(g)
        self.display_list_groups = dict([])
        self.display_list = dict([])
            
    def draw_enable_group(self, group):
        self.draw_set_group(group)
        self.display_list_groups[group] = True

    def draw_disable_group(self, group):
        self.draw_set_group(group)
        self.display_list_groups[group] = False

    def draw_color(self, red=1, green=1, blue=1, group='default'):
        self.draw_set_group(group)
        c = Color(red, green, blue, mode='rgb', group=group)
        self.display_list[group].append(c)

    def draw_point(self, r=0, c=0, group='default'):
        x, y = self.openCVRowColToKivyXY(r, c)
        self.draw_set_group(group)
        p = Point(points=[x,y], group=group)
        self.display_list[group].append(p)

    def draw_line(self, r0=0, c0=0, r1=0, c1=0, group='default'):
        x0, y0 = self.openCVRowColToKivyXY(r0, c0)
        x1, y1 = self.openCVRowColToKivyXY(r1, c1)
        self._draw_line(self, x0=0, y0=0, x1=0, y1=0, group='default')

    def _draw_line(self, x0=0, y0=0, x1=0, y1=0, group='default'):
        self.draw_set_group(group)
        l = Line(points=[x0, y0, x1, y1], group=group)
        self.display_list[group].append(l)
        
    def draw_vector(self, r=0, c=0, angle=0, length=10, arrow=0.2, group='default'):
        points = np.array([[0, length, length, length-length*arrow, length, length-length*arrow],
                           [0, 0,      0,      length*arrow,        0,      -length*arrow],
                           [1, 1,      1,      1,                   1,      1]])
        tmat = np.array([[np.cos(angle), -np.sin(angle), r],
                         [np.sin(angle),  np.cos(angle), c],
                         [0,              0,             1]])
        vecPoints = np.dot(tmat,points)
        pts = self.openCVRowColToKivyXYPts(vecPoints)
        for i in range(0,3):
            self._draw_line(x0=pts[0,2*i], y0=pts[1,2*i], 
                            x1=pts[0,2*i+1], y1=pts[1,2*i+1],
                            group=group)

    def draw_rectangle_centered(self, r=0, c=0, radius=0, group='default'):
        x, y = self.openCVRowColToKivyXY(r, c)
        self.draw_set_group(group)
        r = Line(rectangle=(x-radius, y-radius,2*radius+1,2*radius+1), group=group)
        self.display_list[group].append(r)
        
    def draw_enabled(self):
        for group, enabled in self.display_list_groups.items():
            self.canvas.remove_group(group)
            if enabled:
                for obj in self.display_list[group]:
                    self.canvas.add(obj)


#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

    # function to handle a mouse button pressed event
    def on_touch_up_callback(self, touch):
        # we are only interested in mouse button events
        if 'button' not in touch.profile:
            return
        # we are only interested in left mouse button events
        if (touch.button != 'left'):
            return

        if hasattr(self, 'ud'):
            self.remove_widget(self.ud['label'])
            self.canvas.remove_group(self.ud['group'])

#########################################


