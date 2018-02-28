#lang racket

#|
 $0 is literal 0
 %edi is register in x86 register
 movl from, to
   moves a long value

 lambda_0

 call: pushes the location after the call onto the cpu stack
       jumps to the given location
 nop: is no-operation, like pass in python
 ret: pop a location off the stack, and jump to it

-O3
       movl    $0, %edi
       call    variable
       nop
       ret

-O1

       movl    $10, %edi
       call    variable
       rep ret

movq: move a quad [4 bytes]
%rax is a register
env:   .quad    heap
env(%rip): %rip means top of code in memory when program loads
           means going to env=82 lines after address in %rip

variable(0)
       movq    env(%rip), %rax      // rax = env;
       movq    8(%rax), %rax        // rax = env[1];
       movq    %rax, result(%rip)   // result = rax;
       ret

variable(1) // loop unrolling
       movq    env(%rip), %rax      // rax = env;
       movq    (%rax), %rax         // rax = rax[0];
       movq    8(%rax), %rax        // rax = env[1];
       movq    %rax, result(%rip)   // result = rax;
       ret



       movq    env(%rip), %rdx      // eax = eax - 1
       movl    $9, %eax             // eax = 9

.L18:
       subl    $1, %eax             // eax = eax - 1;
       movq    (%rdx), %rdx         // rdx = rdx[0]
       jne     .L18                 // if that last subtraction wasnt zero jump to .L18
       movq    8(%rdx), %rdx        // rdx = rdx[1]
       movq    %rdx, result(%rip)   // result = rdx
       ret


       movl    $0, %eax
       call    push_result
       nop
       ret

push_result:
       xorl    %eax, %eax           // eax = 0;
       jmp     push_result          // 
|#










