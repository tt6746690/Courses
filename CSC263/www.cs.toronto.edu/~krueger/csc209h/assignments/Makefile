# Example Makefile for 209 a2

# First we define some make variables, to make rules easy to change

CC = gcc
CFLAGS = -Wall -g

# Each target is named; a target can correspond to a file that must be
# generated, or have some other meaning.

# This is the default target (note the ::), and is run if no arguments
# are given to make. Additional targets may be added. This target lists
# no commands to run, and simply asks that the dependencies are up to date.
default:: mycut


# This is the rule to compile mycut. We depend on the .c source file,
# so that if mycut.c changes, make knows that it must recompile the target.
# If mycut is newer that mycut.c, make will realize no recompile is needed.
mycut: mycut.c
	$(CC) $(CFLAGS) $< -o $@

# Note that each command to generate the target must be indented by a
# <tab> character! Spaces don't work! Note the syntax for variable
# substitution. The special variable $@ means target name and $< means 
# list of dependencies (everything after the :). If you add header files
# as dependencies, you'll need to replace the $<.
# Note also that each command is printed by default, so you know what's
# happening.

