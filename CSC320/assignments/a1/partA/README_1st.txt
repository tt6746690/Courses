Assignment #1 
CSC320 Winter 2019
Kyros Kutulakos

Notes on the starter code for the matting application 

---------------
GENERAL REMARKS
---------------

A. REFERENCE SOLUTION BINARY EXECUTABLE

  I am supplying a fully-functional version of the python code in 
  compiled form (ie. a binary, statically-linked executable), so you 
  have a reference solution. I have this for OS X and CDF/Linux:

     viscomp.osx   
     viscomp.cdf/viscomp

  In the case of CDF, the directory viscomp.cdf contains all the 
  libraries that the executable viscomp.cdf/viscomp needs to run.

B. STARTER EXECUTABLE

  The top-level python executable is

     viscomp.py

C. RUNNING THE EXECUTABLES 

  Run viscomp.py --help to see the available command-line options

  The executables operate in two modes:

  1. Triangulation matting 

	 Here is an example run, with the output images stored in tif format:
	
     viscomp.py --matting \
                --backA ../test_images/tiny/flowers-backA.jpg \
                --backB ../test_images/tiny/flowers-backB.jpg \
                --compA ../test_images/tiny/flowers-compA.jpg \
                --compB ../test_images/tiny/flowers-compB.jpg \
                --alphaOut alpha.tif \
                --colOut col.tif

    And here is the same run with the binary executable:
    
    viscomp.cdf/viscomp --matting \
               --backA ../test_images/tiny/flowers-backA.jpg \
               --backB ../test_images/tiny/flowers-backB.jpg \
               --compA ../test_images/tiny/flowers-compA.jpg \
               --compB ../test_images/tiny/flowers-compB.jpg \
               --alphaOut alpha.tif \
               --colOut col.tif
    

  2. Image compositing 

     Here is an example run, with the output images stored in tif format.
     Assuming you've already ran the command above, you can use the
     following
	
	 viscomp.py --compositing \
	            --alphaIn alpha.tif \
	 			--colIn col.tif \
	            --backIn ../test_images/tiny/window.jpg \
	            --compOut comp.jpg

C. TEST IMAGES

     test_images/tiny
     test_images/small
     test_images/large     

       These are directories containing identical
	   images but in different sizes. I suggest you
       begin by using the images in the tiny/
       directory to quickly check things out.

     Each directory contains files of the form

          ????-back?.jpg   An image of the background
       and    
          ????-comp?.jpg   An image of the object in front of that background
     
       as well as images

       window.jpg
       leaves.jpg

     that you can use to create new composites.

---------------------
STRUCTURE OF THE CODE
---------------------

1. GENERAL NOTES

  * partA/viscomp 
       top-level routine that does nothing other than call the
       code's main function, located in partA/matting/run.py

2. IMPORTANT: 

  We will be running scripts to test your code automatically. To 
  ensure proper handling and marking, observe the following:

  * All your code should go in file partA/matting/algorithm.py
  * Do not modify any python files other than algorithm.py
  * Do not modify any parts of algorithm.py except where specified
  * Do not add any extra files or directories under partA/matting/
  * Do not create any extra directories in partA/

3. GENERAL STRUCTURE

  The implementation centers on a single class called
  Matting, defined in algorithm.py. An instance of this
  class is created when the program is first run. It 
  contains private variables that hold all the input and
  output images, methods for reading/writing those
  variables from/to files, and for doing triangulation matting
  and compositing.

3. FILES IN THE DIRECTORY matting/

   algorithm.py	
			You should begin by familiarizing yourself with 
			the Matting class and its available methods. 
			
   run.py	Contains the following two functions
			   main(): 
				  This is the main routine, which parses the 
				  command-line arguments, calls the image-reading
				  and image-writing functions, and runs
				  the triangulation matting or the compositing
				  algorithm. The key lines to look at are
				  l.158 (calling the image-reading method of Matting), 
				  l.187 (calling the image-writing method of Matting),
				  l.171 (running the triangulation matting algorithm)
				  l.171 (running the triangulation matting algorithm)
				  l.223 (image compositing)
			   parseArguments():
			      This routine handles parsing of command-line arguments.
			      You can look at this part of the code to see how the
			      mattingInput(), mattingOutput(),... methods of the Matting
			      class are used, but you do not need to call this function
			      or understand it in detail.

