#lang racket

#| Looking at A1's C in online Compiler Explorer

 x86 Assembly Language
 ---------------------

 $0 is the literal 0
 %edi is register in the x86 cpu
 movl: move a long value
   from, to

 call: pushes the location after the call onto the cpu stack
       jumps to the given location
 nop: is no-operation, like pass in python
 ret: pop a location off the stack, and jump to it
        movl    $0, %edi
          put the argument into register %edi
        call    variable
        nop
        ret

        movl    $10, %edi
        call    variable
        rep ret

  movq: move a quad [8 bytes]
  %rax is a register
  %rip is the program counter

  env: .quad   heap
  env(%rip) : %rip + env
        movq    env(%rip), %rax     // rax = env;
        movq    8(%rax), %rax       // rax = rax[1];
        movq    %rax, result(%rip)  // result = rax;
        ret
  loop unrolling
        movq    env(%rip), %rax  // rax = env;
        movq    (%rax), %rax     // rax = rax[0];
        movq    (%rax), %rax     // rax = rax[0];
        movq    8(%rax), %rax    // rax = rax[1];
        movq    %rax, result(%rip) // result = rax;
        ret

        movq    env(%rip), %rdx // rdx = env;
        movl    $9, %eax        // eax = 9;
.L18:
        subl    $1, %eax        // eax = eax - 1;
        movq    (%rdx), %rdx    // rdx = rdx[0];
        jne     .L18            // if that last subtraction wasn't zero jump to .L18
        movq    8(%rdx), %rax
        movq    %rax, result(%rip)
        ret


        movl    $0, %eax
        call    push_result
        nop
        ret

        xorl    %eax, %eax  // eax = 0;
        jmp     push_result // 
|#