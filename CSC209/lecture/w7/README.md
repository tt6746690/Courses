
__Process Models__


+ _program_: executable instruction of program
  + source code
  + compile machine code
+ _process_: running instance of a program
  + machine code of program
  + current state info
    + stored in ![](assets/README-70115.png)
    + _stack_
      + what function is executing / holds values for local variables
    + _global variable_ and _heap_
      + holds other variables
    + _OS_
      + keeps track of additional state for process
      + ![](assets/README-cabd8.png)
        + _Process Control Block_: data structure for process
          + `PID`: process identifier
          + `SP`: stack pointer (points top of stack)
          + `PC`: program counter (identify next instruction to execute)




__`Top`__
+ displays active processes

![](assets/README-e2fff.png)
+ `PID`
+ Command executing


Number of CPU determines how many processes can be executing instruction at same time
![](assets/README-93455.png)

Process undergo state changes; OS scheduler decides when processes are in which state
+ running state
+ ready state
+ blocked state
  + waiting for an event to occur, i.e.
    + made read call and waiting for data to arrive
    + called sleep waiting for timer to expire


![](assets/README-2a90c.png)
+ `gcc` move to blocked state because want to access a file
+ once arrived, `gcc` moves to ready state
+ once a processor is available, `gcc` moves to running state


![](assets/README-23342.png)


---


__Creating processes__

`fork`
+ creates a new child process by duplicating the calling process
  + two processes does not share memory
  + OS determines which is ran first
+ returns
  + 0 to child process
  + process ID of child process to parent process
+ returns -1 to parent process if child process is not successfully created

![](assets/README-5d628.png)
+ comparison
  + Same
    + code
    + variable
    + `PC`
      + hence child process starts running after `fork` returns
    + `SP`
  + Different
    + `PID`
    + return value

![](assets/README-175db.png)


![](assets/README-35180.png)


---


__process relationship and termination__
+ creates 5 processes
+ each child processes go through a loop with `usleep`, which passes control to OS

![](assets/README-389f3.png)
+ single process in the correct order.
+ different processes are not ran in a predictable way
+ parent process terminates before child processes finished to print
  + it finishes regardless of child processes
  + the shell prompt ![](assets/README-2dc11.png)
  + note shell waits for current (parent) process to finish and prints a prompt for next command
    + program running from shell is a child of the shell process.
    + uses `wait` sys call to suspend itself until its child terminates

![](assets/README-ed23c.png)



__`pid_t wait(int *stat_loc);`__
+ suspends execution of its calling process until `stat_loc` info available for a terminated child process (i.e. one of the child terminates)
+ returns
  + child PID if child process termiates
  + -1 if wait fails
+ `stat_loc`
  + pointer to termination status
    + lower 8 bit -> termination status
    + next 8 bit -> value returned from `exit`
        + `exit(0)` - successful
        + `exit(1)` (non zero) - failure
          + `stat_loc` is 256 in this case
  + extract useful info
    + with bitwise `|`
    + or with macros
      + ![](assets/README-30edf.png)

![](assets/README-3c9d5.png)
+ will be in block state until all child processes terminates


![](assets/README-bbd11.png) ![](assets/README-482d6.png)
+ `abort()` child process number 2
  + returns the proper signal 6 for `abort`


`pid_t waitpid(pid_t pid, int *stat_loc, int options);`
  + if `pid = -1` will wait for any child process
  + if `pid > 0` waits for child process with `pid`

![](assets/README-797dd.png)
+ both `wait` and `waitpid` waits for direct child processes
+ `status` contains both return code from terminating child as well as signals


__W7 exercise__

![](assets/README-acb79.png)
+ a loop for creating child processes

![](assets/README-0258e.png)
+ wait for a chain of child processes to terminate

![](assets/README-180c0.png)
+ wait for a group of child from a single parent to terminate

---


__Zombies and Orphans__

> what happens when child process terminates before `wait` is called


__Zombie__
  + dead process (terminated) that still hangs around waiting for parent to collect its termination status
  + `top`
    + terminated child process has a state of -> `Z` for zombie and `<defunct>`
    + sys cannot delete process control block of terminated process in case the parent calls `wait` later
  + exercised up after termination status is collected


> what happens to child process when its parent terminates naturally?

![](assets/README-b1e1e.png)
+ the child process is called an __orphan__ when the parent process terminates first
  + the orphans has `getppid()` return value of 1, the `init` process (the first process OS launches)
    + `orphans` adopted by `init`
    + `init` calls `wait` in a loop to collect termination status of any process that it has adopted
      + after which a process's data structure can be deleted and the zombie disappears

![](assets/README-60f78.png)

![](assets/README-402b2.png)
+ process not necessarily removed from process table when it terminates (zombie)

---

__Running different programs__

`int execl(const char *path, const char *arg0, ... /*, (char *)0 */);`
+ _replaces_ currently running process with a new process
  + note `exec` modifies the calling process and does not create new processses.
    + `PID` is same
    + _pipe_ retains original open file descriptors
  + loads executable to into memory where code segment is. and initializes a new stack; `PC` and `SP` are updated to execute the new process
  + the original code is _GONE_ so should never execute return here
    + ![](assets/README-33e82.png)
+ will return
  + if error occur, i.e. could not load the program



`exec` family
suffixes
+ __l__ a list of arguments are passed to `exec`
  + ![](assets/README-48cd5.png)
+ __v__ an array of string are passed to `exec`
  + ![](assets/README-67ef9.png)
+ __p__ PATH environment variable is used to search executable
  + ![](assets/README-42a30.png)
+ __empty__ full path must be provided
  + ![](assets/README-c8951.png)
+ __e__ array of env variables
  + ![](assets/README-116ff.png)


__Shell__
+ a process that uses `fork` and `exec`
+ steps
  + upon entering command, use `fork` to create child process
  + use `exec` to load different program into the memory of child process
  + calls `wait` and blocks until child process finishes executing
  + when `wait` returns, prints a prompt.


![](assets/README-573a9.png)
+ note `printf` might execute if `exec` fails
