

# 2.4 Scheduling 


+ _definitions_
    + _scheduling_ 
        + determine which of the two or more of processes/threads in _ready_ state gets to run next
        + considers making efficient use of CPU to avoid context switching
    + _scheduler_
        + part of OS that makes such choice 
    + _scheduling algorithm_
        + algorithm it uses 
+ _intro_ 
    + _history_
        + batch system: run next job 
        + batch and timesharing service: decide if a batch job or interactive user should go next 
        + PC: 
            + changes: just one user present and CPU is not a scarse resource 
            + hence not of importance
        + networking server: decide if a process that gathers daily stat vs. one that handles user request 
    + _process behavior_    
        + ![](2017-06-18-18-23-14.png)
        + nearly all processes alternate bursts of _computing_ with _(disk or network) I/O_
            + e.x. CPU runs for a while, syscall made to read a file. When syscall finishes, CPU computes again
        + _Compute-bound (CPU-bound)_: 
            + spend most of their time computing 
            + long CPU bursts and infrequent I/O waits 
        + _I/O bound_: 
            + spend most their time waiting for I/O
            + short CPU bursts and frequent I/O waits 
            + should prioritize I/O requests so as to make disk busy
        + _observation_ 
            + _length of CPU bursts_ is of matter here, not _length of I/O waits_ 
                + IO-bound processes are IO bound because they do not compute much between IO requests, but because they have especially long I/O requests 
            + processes tend to get more I/O-bound as CPU gets faster (i.e. CPU burst gets shorter in length)
                + because CPU improving faster than disks 
    + _when to schedule_ 
        + _situations_ 
            + _`fork`_: pick either child or parent, both in _ready_ state 
            + _`exit`_: an idle process is run 
            + _blocked on I/O, on semaphore, ..._: 
            + _I/O iterrupt_: run the process that waited for the I/O job to finish, or some other process 
        + _clock interrupts_: 
            + hardware clock provides periodic interrupts at 60Hz or other frequency, a scheduling decision can be made at each or `k`th interrupts
            + _nonpreemptive scheduling algo_: picks a process to run and then just lets it run until it blocks (on I/O or waiting for another process) or voluntarily release the CPU
                + can theoretically run forever 
                + no scheduling decision at clock interrupts 
            + _preemptive scheduling algo_: picks a process and lets it run for a maximum of some fixed time.If it is still running at end of time interval, it is suspended and the scheduler picks another process to run 
                + requires clock interrupt at end of time interval to give control of CPU back to scheduler
    + _categories of scheduling algo_ 
        + different algo is needed since different application area have different goals 
        + _environment_ 
            + _batch_ 
                + for periodic tasks with no interactive users 
                + nonpreemptive algorithms, or preemptive algorithm with long time periods for each process, are acceptable
                    + reduces process switches and improves performance
            + _interactive_ 
            + _real time_ 
        
                




