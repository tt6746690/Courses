@echo off

REM paths
DOSKEY course = cd "C:\000 My Stuff ######\School\2015-2016 courses"
DOSKEY csc148 = cd "C:\000 My Stuff ######\School\2015-2016 courses\CSC148"

REM basics
REM $T = & ,used if there is more than 2 commands in a DOSKEY macro
REM DOSKEY cdd=cd $1$Tdir /B
DOSKEY ls=dir /B
DOSKEY cp=copy $*
DOSKEY mv=move $*
DOSKEY rm=del /p $*
DOSKEY rmf=del /q $*
REM rmdir /s /q
DOSKEY clear=cls

REM applications
DOSKEY n=notepad $*
DOSKEY e=explorer $*

REM utils
DOSKEY touch=FSUTIL file createnew $* 0
DOSKEY fs= FIND /i $* ./*
DOSKEY tasklist= TASKLIST /FO TABLE


REM doskey stuff
DOSKEY macros=DOSKEY /macros
DOSKEY h=DOSKEY /history

REM system info
Doskey getBios=WMIC BIOS Get Manufacturer,Name,Version /Format:csv
Doskey patches=WMIC qfe list full /format:htable
Doskey version=WMIC csproduct get Version 
