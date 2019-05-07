Assignment #2
CSC320 Winter 2019
Kyros Kutulakos

Notes on the starter code for the inpainting application 

---------------
GENERAL REMARKS
---------------

A. REFERENCE SOLUTION BINARY EXECUTABLE

  I am supplying a fully-functional version of the python code in 
  compiled form (ie. a binary, statically-linked executable), so you 
  have a reference solution. 
  
B. STARTER EXECUTABLE

  The top-level python executable is

     viscomp-gui.py

C. RUNNING THE EXECUTABLES 

  To run, use
     python viscomp-gui.py -- --usegui

D. GETTING FAMILIAR WITH THE INTERFACE

  1. The lower-right button shows the GUI's "current image". Clicking
     on that button will open a dialog box to load that image from a file
     (if it is an input image) or to save to a file (if the image is an
     output image of the algorithm).
  2. Click on the "Switch Image" button to cycle
     through the algorithm's input and output images.
  3. Clicking with the left mouse button on the image being displayed
     shows a pair of red axes centered on the point being clicked, along
     with the point's coordinates. These axes disappear after the mouse
     button is released.
  4. Dragging the mouse button drags the image. The image can be moved 
     back to its original position by double-clicking/double-tapping on
     the image.
  5. The "Switch Modes" button doesn't do anything so it can be ignored.
  6. Pressing the escape key closes the GUI and terminates the program
  7. You will need to choose a color source image and a
     binary alpha matte (represented as a single-channel uint8 image)
     of the same size as the source.
  8. Click on the Debug button to see the controls you have over the
     algorithm. I suggest you leave these to their default settings
     for the first few times you run the algorithm. Pressing ESC
     makes the popup window disappear.
  9. After loading the mask the the source image, press the Step 
     button to execute one iteration of the inpainting algorithm.
     You should see the following: 
        (a) A blue box centered on the boundary of the unfilled
            region. This is the current patch, psi_p
        (b) A green box that is fully contained in the known part of
            the image. This is the patch psi_q whose pixels are most
            similar to the filled pixels of psi_p.
        (c) The green vector indicates the normal to the
            fill front curve at the green patche's center, computed
            by the computeNormal() function.
        (d) The red vector indicates the gradient assigned to psi_p
            by the computeGradient() function.
 10. Click the "Switch Image" button to cycle through the various
     intermediate images created by the algorithm.
 11. To get a better feel for how the algorithm works, it is best
     to press the Step button while the Inpainted Image is being
     shown. This will allow you so see how similar the green and
     blue patches really are, and to verify that gradient vector
     and fill front normals are reasonable.
 12. Now press Step a few times while viewing the fill front, the
     confidence image, etc. This will give you a visual idea of
     how these images are being modified.
 13. The default mode is to have all debugging information shown.
     This includes showing in the terminal window the actual
     contents of the various patches used by the algorithm. You 
     will need to pay close attention to this data (esp the 
     fill front and psiHatP) since this is the data upon which
     your vector calculations take place.
 14. Leaving only verbose output enabled will show just the
     iteration number and some basic information about the 
     patch being selected. This is probably the best setting
     for pressing the "Run" button which will go through many
     iterations. Moving the "Run Iterations" scrollbar to the
     leftmost position in the Debug Control popup will set the
     iterations to -1, which indicates that the algorithm will
     run until all pixels have been inpainted.
 15. You can change the patch radius to experiment with different
     patch sizes. This will definitely have an impact on both the
     result and the execution time.

E. EXTENDING THE STARTER CODE

  1. The starter code is (almost) ready to run the inpainting
     algorithm: once you transfer your image-loading function from A1
     into file algorithm.py and implement the Run button you should 
     be able to load images and run iterations of the algorithm. 
  2. The only problem with the starter code as-is is that the
     patch priorities are not computed correctly because it uses
     dummy values for the gradient, normal, and confidence. So 
     you are likely to get poor inpainting results out of the box.
  3. I suggest you start by implementing the computeC() function
     which should improve the inpainting result quite a bit already.
     After that, implement the other two functions so that priorities
     can be computed according to the algorithm in the paper.


---------------------
STRUCTURE OF THE CODE
---------------------

1. GENERAL NOTES

  * code/viscomp-gui
       top-level routine that does nothing other than call the
       code's main function, located in code/inpaintingui/run.py

2. IMPORTANT: 

  We will be running scripts to test your code automatically. To 
  ensure proper handling and marking, observe the following:

  * All your widget specs should go in file code/kv/viscomp.kv 
  * All your UI-related code should go in the files code/inpaintingui/widgets.py
    and code/inpaintingui/viewer.py 
  * All your inpainting-related code should go in the following files:
  		code/inpainting/compute.py 
        code/inpainting/algorithm.py 
        code/inpaintingui/viewer.py
  * The only modification you will do to algorithm.py is to transfer the 
    functionalities you implemented in A1 as these
    are not provided in the A2 starting code either.
  * Do not modify any other python files
  * Do not modify any parts of the above files except where specified
  * Do not add any extra files or directories


3. GENERAL STRUCTURE -- GUI

  The GUI is defined in the file kv/viscomp.kv. This defines the entire
  set of widgets used by the GUI and controls which methods are called
  in response to various GUI events (button presses, mouse clicks, etc).
  You need to start by reading this file. It is heavily commented, to 
  guide you through its structure, etc. ** you will need to add a few specs
  to this file in order to implement the functionalities requested in A2

  Each Kivy widget is an instance of a Kivy widget class. The most important
  class in the code is RootWidget. This class is defined in file 
  mattingui/widgets.py. Read this file next and try to understand it well. 
  You will need to write one of its methods.

  To load/save images in the Inpainting object you used in A1, the code uses
  a class called InpaintingControl. This 'sits' in between the RootWidget class
  and the Inpainting class. The RootWidget class you need to write will have to
  call one of the methods in this file.

  The last file you should look at is viewer.py. This file defines a widget
  called ImageViewer that controls how images are displayed and how a user
  can interact with those images. You will need to implement a method in this
  class as part of the assignment, so read this code carefully. 

4. GENERAL STRUCTURE -- Inpainting algorithm

  The implementation centers on a single class called
  Inpainting, defined in algorithm.py. An instance of this
  class is created when the program is first run. It 
  contains private variables that hold all the input and
  output images, methods for reading/writing those
  variables from/to files, and for doing triangulation matting
  and compositing.

  The next most important class is PSI, which is used to represent
  patches. This is defined in psi.py. You will need to get very 
  familiar with this class. The functions in copyutils.py may also 
  be useful (although most of their functionality is encapsulated
  in the PSI class).

5. FILES IN THE DIRECTORY code

   UI-related files:
   
   kv/viscomp.py	
			Specifications written in the kivy widget specification language.
			The comments in this file should be sufficient to understand its
			basic structure. The assignment does NOT depend on learning much
			about this language beyond what is in this file already.

   inpaintingui/widgets.py
			The only relevant part of this file are the methods of the 
			RootWidget class. 
			
   inpaintingui/control.py
		    The only relevant functions are those labelled "Top-level methods 
            when interacting with the GUI". You can ignore the rest, 
            at least initially.

   inpaintingui/viewer.py
			A widget class for displaying images. This is completely
			independent of the inpainting algorithm. You will need to add 
			(at least) one method to this class for this assignment. 

   Inpainting-related files:

   inpainting/compute.py
            To understand the inpainting algorithm, start from here 
            after reading the paper and running the reference and/or 
            starter implementation. This file contains 
            the skeleton of the 3 main functions you must implement.
  
   inpainting/algorithm.py	
			This file contains the bulk of the inpainting algorithm,
			including its main loop (from Table 1 of the paper). 
			Take a few minutes to add the code snippets from A1 so
			you can run the start code. You should also browse the
			method inpaintRegion() which contains the algorithm's
			basic loop.
			
   inpainting/psi.py, inpainting/copyutils.py
            The methods in these  files are the ones you are likely to 
            be the most important for your implementation, so take the
            time to read and understand them.
    
   inpainting/debug.py, inpainting/patchdb.py
            You do not need to look at these files

      
