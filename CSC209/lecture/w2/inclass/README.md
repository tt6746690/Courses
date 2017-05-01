

A `_wx` file
  + the file cannot be read by commands like `cat`
  + if the file requires compilation to execute, it cannot be executed even if `x` permitted.
    + i.e. `gcc -o hello_world.c` permission denied
  + if the file is compiled machine code, it can be executed because dont need to read it first
    + i.e. `hello_world`


So if the parent file directory has `_wx`
  + no read on directory
    + basically cannot read which files are in there
    + i.e. cant tab completion the file within the directory
    + i.e. cant `ls` within the directory although can `cd` in.
  + but can execute file within directory if it has `x` execute




__wednesday__

+ `:! gcc -o filename % && filename.c` in vim compiles

+ `#define SIZE 100` is a proprocessor which you cannot change, the compiler replace the code with value assigned.

memory
+ `int` takes up 4 bytes; `char` takes up 1 byte.
+ 1 byte the smallest addressable unit of memory in many computer architectures.


```c
int a[4];
int i;
for(i=0; i< 100000; i++){
  a[i] = i
}
// gives segmentation fault error. => accessing memory that is off permission
```
