
symlink
+ `->` links are aliases, they contain path 


hardlink 
+ both filenames refer to the same data on disk
+ directories cannot have hard links 
    + so that every directory has only one parent

directories 
+ cannot have hard links 
+ have just one parent 
    + `.` and `..` access current and parent directory 


filesystem 
+ OS module consists of data structure and operations for files on disk


```c 
#ifndef <token>
/* code */
#else
/* code to include if the token is defined */
#endif
```
+ check whether token has been `#define` earlier
