## CSC320 Winter 2019 
## Assignment 1
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE)
## WITHOUT WRITTEN AUTHORIZATION OF THE INSTRUCTOR IS STRICTLY 
## PROHIBITED. VIOLATION OF THIS POLICY WILL BE CONSIDERED AN
## ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY ANY PART OF THIS FILE
##

# import basic packages
import os
import sys
import argparse
import cv2 as cv
import numpy as np
from algorithm import Matting

# Routine for parsing command-line arguments
# Upon success, it returns any unprocessed/unrecognized arguments 
# back to the caller
#
# Input arguments
#   - argv:   input command line arguments (as returned by sys.argv[1:])
#   - mat:    instance of the Matting class, which holds all the 
#             data needed to run the triangulation matting algorithm
#             and to create image composites
#   - prog:   name of the executable (as returned by sys.argv[0])
# Return values
#   - True if all the required arguments are specified and False otherwise
#   - a Namespace() object containing the user-specified arguments and
#     any optional arguments that take default values
#   - a string containing any unrecognized/unprocessed command-line
#     arguments
#   - an error message (if success=False)
#     
def parseArguments(argv, mat, prog=''):
    # Initialize the command-line parses
    parser = argparse.ArgumentParser(prog)
    
    # The two main arguments for controlling program execution are
    #    --matting for computing object color and alpha from 
    #      backgrounds & composites
    #    --compositing for computing a new composite given color, 
    #      alpha and background
    #
    # We use the following statement to ensure exactly one of them 
    # appears on the command line.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--matting', \
                        default=False, \
                        action='store_true', \
                        help='Run triangulation matting algorithm')
    group.add_argument('--compositing', \
                        default=False, \
                        action='store_true', \
                        help='Create a composite')
    
    # The rest of the command-line arguments are obtained by 
    # querying the matting object for the input and output
    # filenames. See algorithm.py for details on the methods
    # mattingInput(), mattingOutput(), compositingInput(), 
    # compositingOutput()
    for a in mat.mattingInput().keys():
        parser.add_argument('--%s'%a, nargs=1, \
                        default=mat.mattingInput()[a]['default'], \
                        help=mat.mattingInput()[a]['msg'])
    for a in mat.mattingOutput().keys():
        parser.add_argument('--%s'%a, nargs=1, \
                        default=mat.mattingOutput()[a]['default'], \
                        help=mat.mattingOutput()[a]['msg'])
    #
    for a in mat.compositingInput().keys():
        parser.add_argument('--%s'%a, nargs=1, \
                        default=mat.compositingInput()[a]['default'], \
                        help=mat.compositingInput()[a]['msg'])
    for a in mat.compositingOutput().keys():
        parser.add_argument('--%s'%a, nargs=1, \
                        default=mat.compositingOutput()[a]['default'], \
                        help=mat.compositingOutput()[a]['msg'])

    # Run the python argument-parsing routine, leaving any
    # unrecognized arguments intact
    args, unprocessedArgv = parser.parse_known_args(argv)

    # Check that the correct arguments are specified for each of the
    # two modes (--matting and --compositing)
    #
    # Note: The parser only checks that the specified filenames
    #       exist and/or are readable/writeable as needed
    #       they do *not* check the consistency of the actual images
    #       (eg that all input images have the same dimension). 
    #       Those checks should be performed by methods in the 
    #       matting class
    
    success = True
    msg = ''
    if (args.matting is None) and (args.compositing is None):
        success = False
        msg = 'one of --matting and --compositing must be specified'
    if args.matting:
        if (args.backA is None) or (args.backB is None) or \
           (args.compA is None) or (args.compB is None):
            success = False
            msg = 'images backA, backB, compA, compB must be specified in matting mode'
    elif args.compositing:
        if (args.colIn is None) or (args.alphaIn is None) or \
           (args.backIn is None):
            success = False
            msg = 'images colIn, alphaIn, backIn must be specified in compositing mode'
            
    # return any arguments that were not recognized by the parser
    return success, args, unprocessedArgv, msg

#
# Top-level routine for performing triangulation matting and 
# compositing when running the code from the command line
#
# Input arguments
#   - argv:   list of command-line arguments (as returned by sys.argv[1:])
#   - prog:   name of the executable (as returned by sys.argv[0])
# Return value
#   - list that holds any unrecognized command-line arguments
#     (in the same format as sys.argv[1:])

def main(argv, prog=''):
    # Create an instance of the matting class
    mat = Matting()
    # Parse the command line arguments
    success, args, unprocessedArgv, msg = parseArguments(argv, mat, prog)

    if not success:
        print 'Error: Argument parsing: ', msg
        return success, unprocessedArgv
    
    # Initialize the variables for measuring timings
    t1 = t2 = t3 = t4 = 0
    if args.matting:
        # Get the value of the openCV timer
        t1 = cv.getTickCount()
        
        # Call the routine that reads the images and stores them in 
        # the appropriate data structure stored with the matting 
        # instance
        #
        # Note: The images themselves are supposed to be private 
        # variables. They should be acccessed using the 
        # readImage(), writeImage(), useTriangulationResults() 
        # methods of the matting class. The images are referenced 
        # by their string descriptor
        for (fname, key) in [(args.backA[0], 'backA'), 
                             (args.backB[0], 'backB'), 
                             (args.compA[0], 'compA'),
                             (args.compB[0], 'compB')]:
            success, msg = mat.readImage(fname, key)
            if not success:
                print 'Error: %s'%msg
                return success, unprocessedArgv

        # Get the value of the openCV timer
        t2 = cv.getTickCount()
        
        # Run the triangulation matting algorithm. The routine
        # returns a tuple whose first element is True iff the routine
        # was successfully executed and the second element is
        # an error message string (if the routine failed)
        print 'Triangulation matting...'
        success, text = mat.triangulationMatting()
        if not success:
            print 'Error: Triangulation matting routine failed: %s'%text
            return success, unprocessedArgv

        # Get the value of the openCV timer
        t3 = cv.getTickCount()
        
        # Call the routine that writes to disk images stored in a matting
        # instance.
        # Note: The images themselves are supposed to be private variables
        #       They should be acccessed using the readImage(), writeImage()
        #       and useTriangulationResults() methods of the matting class
        #       The images are referenced by their string descriptor 
        for (fname, key) in [(args.colOut[0], 'colOut'), 
                             (args.alphaOut[0], 'alphaOut')]:
            success, msg = mat.writeImage(fname, key)
            if not success:
                print msg
                print 'Error: Image %s cannot be written'%fname
                return success, unprocessedArgv

        
        # Get the value of the openCV timer
        t4 = cv.getTickCount()

    elif args.compositing:
        t1 = cv.getTickCount()
        
        # Call the routine that reads the images and stores them in 
        # the appropriate data structure stored with the matting 
        # instance
        #
        # Note: The images themselves are supposed to be private 
        # variables. They should be acccessed using the 
        # readImage(), writeImage(), useTriangulationResults() 
        # methods of the matting class. The images are referenced 
        # by their string descriptor
        for (fname, key) in [(args.colIn[0], 'colIn'), 
                             (args.alphaIn[0], 'alphaIn'), 
                             (args.backIn[0], 'backIn')]:
            success, msg = mat.readImage(fname, key)
            if not success:
                print 'Error: %s'%msg
                return success, unprocessedArgv
        
        print 'Compositing...'
        t2 = cv.getTickCount()
        # Run the compositing algorithm. The routine
        # returns a tuple whose first element is True iff the routine
        # was successfully executed and the second element is
        # an error message string (if the routine failed)
        success, text = mat.createComposite()
        if not success:
            print 'Error: %s'%text
            return success, unprocessedArgv

        t3 = cv.getTickCount()
        success, msg = mat.writeImage(args.compOut[0], 'compOut')
        if not success:
            print msg
            print 'Error: Image %s cannot be written'%fname
            return success, unprocessedArgv
        t4 = cv.getTickCount()

    print '----------------------------------\nTimings\n----------------------------------'
    print 'Reading:    %g seconds'%((t2-t1)/cv.getTickFrequency())
    print 'Processing: %g seconds'%((t3-t2)/cv.getTickFrequency())
    print 'Writing:    %g seconds'%((t4-t3)/cv.getTickFrequency())
    
    # return any command-line arguments that were not processed
    return success, unprocessedArgv


# Include these lines so we can run the script from the command line
if __name__ == '__main__':
    runMatting(sys.argv[1:], sys.argv[0])

