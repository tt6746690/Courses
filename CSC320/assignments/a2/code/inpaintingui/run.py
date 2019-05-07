## CSC320 Winter 2019 
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY


##
## DO NOT MODIFY ANY PART OF THIS FILE
##

import sys
import argparse


# import the UI-related modules
from inpaintingui import viewer
from inpaintingui.widgets import VisCompApp


def main(argv, prog=''):
    VisCompApp().run()
        

if __name__ == '__main__':
    main(sys.argv[1:], sys.argv[0])
