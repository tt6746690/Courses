[BATCH SCRIPTING](http://steve-jansen.github.io/guides/windows-batch-scripting/)
===============


### Getting started

batch file is a script file that consists of a series of commands to be executed by the command line interpreter, stored in plain text file ending in `.cmd` or `.bat`

#### basics

+ comments
`REM` is the Remark keyword

```bash
REM this is a comment

:: this is a comment too
```

+ silencing display of commands in batch files  
the first non-comment line usually a command to turn off ECHO of each batch file line. Here @ suppresses echo  

```bash
@ECHO OFF
```

To restore printing of commands

```bash
ECHO ON
```


### Variables

Declaration  
DOS does require declaration of variables. Values of uninitialized variable is an empty string, or `""`.  

Assignment  
use `SET` assign a value to variable. `/A` supports arithmetic operations. Since `SET` always override existing variables. A quick echo will confirm that the variable is not an existing system variable

```bash
SET foo=bar   # note that foo = bar will not work!

SET /A four=2+2
4
```

Reading  
Read the value of a variable by prefixing and postfixing variabel name with `%`. Use `SET` simply to check all system-wide environment variable, like `%PATH%`, `%TEMP%`, `%DATE%`.

```bash
C:\> SET foo=bar
C:\> ECHO %foo%
bar
```

Variable scope  
  + by default, variables are Global
  + to set local variables, use `SETLOCAL`
  + revert varaible assignemnt siwh `ENDLOCAL`
  + application would be to modify `%PATH%` in batch

```bash
#test.cmd
SETLOCAL
SET V=Local Vale
ENDLOCAL

# terminal
:: $ SET v=Global Value
:: test.cmd
:: $ ECHO v=%v%
:: v=Global Value
```

Commandline arguments     
use `%[1-9]` to access commandline arguments passed onto the scripts.

```bash
# run.cmd
ECHO %0   # run.cmd
ECHO %1   # value of the first argument
ECHO %~1  # ~ removes quotes (usefor for pathname)

ECHO %~f1     # full path to directory of first command line argument, a path
ECHO %~fs1    # shorter version of ~f1 DOS8.2

ECHO %~dp1    # full path to parent directory of first command line argument
SET parent=%~dp0   

ECHO %~nx1    # file name and file extension of first command line argument
ECHO %~n0: some message
```

Some very useful starter code

```bash
SETLOCAL ENABLEEXTENSIONS   # command processor extension
SET me=%~n0                 # name of the script, without extension
SET parent=%~dp0            # parent of the script
```

### Return Codes

Checking return value   
The environmental variable `%ERRORLEVEL%` contains the return code of the last executed program or script. A very helpful feature is the built-in DOS commands like `ECHO`, `IF`, and `SET` will preserve the existing value of` %ERRORLEVEL%`.


```bash
IF %ERRORLEVEL% NEQ 0 (       # check for non zero return code
  REM do something here to address the error
)

# SomeFile.exe
IF %ERRORLEVEL% EQU 9009 (
  ECHO error - SomeFile.exe not found in your PATH
)
```

Conditional Execution Using the Return Code   
Execute a second command based on the success or failure of the first command.The first program/script must conform to the convention of returning 0 on _success_ and non-0 on _failure_ for this to work.

```bash
# To execute a follow-on command after sucess, we use the `&&` operator
SomeCommand.exe && ECHO SomeCommand.exe succeeded!

# To execute a follow-on command after failure, we use the || operator
SomeCommand.exe || ECHO SomeCommand.exe failed with return code %ERRORLEVEL%
```

By default, the command processor will continue executing when an error is raised. `&&` and `||` can be used to halt program should error arise. The `EXIT` command with `/B` switch will exit the current batch script context, and not the command prompt process. A simliar technique uses the implicit `GOTO` label called `:EOF` (End-Of-File). Jumping to EOF in this way will exit your current script with the return code of 1.

```bash
SomeCommand.exe || EXIT /B 1
SomeCommand.exe || GOTO :EOF
```

Tips
Stick to `0` with success and positive values for failures

```bash
SET /A ERROR_HELP_SCREEN=1
SET /A ERROR_FILE_NOT_FOUND=2
SET /A ERROR_FILE_READ_ONLY=4
SET /A ERROR_UNKNOWN=8
```

### Stdin, stdout, stderr  
DOS, like Unix/Linux, uses the three universal “files” for keyboard input, printing text on the screen, and the printing errors on the screen. The “Standard In” file, known as stdin, contains the input to the program/script. The “Standard Out” file, known as stdout, is used to write output for display on the screen. Finally, the “Standard Err” file, known as stderr, contains any error messages for display on the screen.  


File Number  
Each of these three standard files, otherwise known as the _standard streams_, are referernced using the numbers 0, 1, and 2. Stdin is file 0, stdout is file 1, and stderr is file 2.

Redirection  
A very common task in batch files is sending the output of a program to a log file. The `>` operator sends, or redirects, stdout or stderr to another file. Use `<` to use content of file as input. Use `|` for chaining, or piping.

```bash
DIR > temp.txt    # write a list of file name to a text file
DIR >> temp.txt   # append instead

DIR SomeFile.txt  2>> error.txt   # redirect stderr (> and >> defaults to stdin, stdout)

DIR SomeFile.txt > output.txt 2>&1    # write stdout/stderr to the same file

# SORT sort input file according to alphabet
SORT < SomeFile.txt     # use content of file as input, instead of using argument

PING 127.0.0.1 > NUL    # NUL discard any output from program

DIR /B | SORT   # sorting the output of DIR command using PIPE
```



### If/Then conditionals    


#### Examples   

```bash
# check for file existence
IF EXIST "temp.txt" ECHO found

# or converse
IF NOT EXIST "temp.txt" ECHO not found

# together
IF EXIST "temp.txt" (
    ECHO found
) ELSE (
    ECHO not found
)


# Check if a variable is set
IF "%var%"=="" (SET var=default value)    # defults of %foo% is empty string
IF NOT DEFINED var (SET var=default value)

# check if variable match string
SET var=Hello, World!
IF "%var%"=="Hello, World!" (
    ECHO found
)

# case insensitive comparison
IF /I "%var%"=="hello, world!" (
    ECHO found
)

# arithmatic comparison
SET /A var=1
IF /I "%var%" EQU "1" ECHO equality with 1
IF /I "%var%" NEQ "0" ECHO inequality with 0
IF /I "%var%" GEQ "1" ECHO greater than or equal to 1
IF /I "%var%" LEQ "1" ECHO less than or equal to 1

# checking return code
IF /I "%ERRORLEVEL%" NEQ "0" (
    ECHO execution failed
)
```



### Loops
Looping through items in a collection is a frequent task for scripts. It could be looping through files in a directory, or reading a text file one line at a time.

GOTO  
The old-school way of looping on early versions of DOS was to use labels and GOTO statements. This isn’t used much anymore, though it’s useful for looping through command line arguments.

```bash
:args
SET arg=%~1
ECHO %arg%
SHIFT
GOTO :args
```

FOR   
The FOR command uses a special variable syntax of % followed by a single letter, like %I. This syntax is slightly different when FOR is used in a batch file, as it needs an extra percent symbol, or %%I. This is a very common source of errors when writing scripts. Should your for loop exit with invalid syntax, be sure to check that you have the %% style variables.

```bash
# looping through files
FOR %I IN (%USERPROFILE%\*) DO @ECHO %I

# looping through directories
FOR /D %I IN (%USERPROFILE%\*) DO @ECHO %I

# Recursively loop through files in all sub-directories
FOR /R "%TEMP%" %I IN (*) DO @ECHO %I

# Recursively loop through all sub-directories
FOR /R "%TEMP%" /D %I IN (*) DO @ECHO %I
```



### Functions
Functions are de facto way to reuse code in just about any procedural coding language. While DOS lacks a bona fide function keyword, you can fake it till you make it thanks to labels and the CALL keyword.

+ Your quasi functions need to be defined as labels at the bottom of your script.  
+ The main logic of your script must have a EXIT /B [errorcode] statement. This keeps your main logic from falling through into your functions.

```bash
# poor-man's version of tee, write message to file/stdout
@ECHO OFF
SETLOCAL

:: script global variables
SET me=%~n0
SET log=%TEMP%\%me%.txt

:: The "main" logic of the script
IF EXIST "%log%" DELETE /Q %log% >NUL

:: do something cool, then log it
CALL :tee "%me%: Hello, world!"     # CALL invoke lable :tee (can pass argument)

:: force execution to quit at the end of the "main" logic
EXIT /B %ERRORLEVEL%

:: a function to write to a log file and write to stdout
:tee
ECHO %* >> "%log%"    # append all arguments to log
ECHO %*       # but can use ECHO to print to stdout, allow for downstream pipe
EXIT /B 0     # no other way to return code other than exit code
```


### Parsing
Robust parsing of command line input separates a good script from a great script. I’ll share some tips on how I parse input. ffBy far the easiest way to parse command line arguments is to read required arguments by ordinal position.

```bash
# full path passed to script as first argument
SET filepath=%~f1

# raise error if the file does not exist
IF NOT EXIST "%filepath%" (
    ECHO %~n0: file not found - %filepath% >&2
    EXIT /B 1
)
```


### Logging
Logging helps with troubleshooting both during execution and after execution.

### Tips   


Conventions    

```bash
:: Name:     MyScript.cmd
:: Purpose:  Configures the FooBar engine to run from a source control tree path
:: Author:   name
:: Revision: March 2013 - initial version
::           April 2013 - added support for FooBar v2 switches

@ECHO OFF
SETLOCAL ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

:: variables
SET me=%~n0


:END
ENDLOCAL
ECHO ON
@EXIT /B 0
```

```bash

```
