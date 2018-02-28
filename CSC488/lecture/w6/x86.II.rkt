#lang racket
; (%rip) - the bracket notation is like dereference in C

; void push(void* v) { top[0] = v; top += 1; }

;         movq    top(%rip), %rax    // rax = top
;         movq    %rdi, (%rax)       // rax[0] = rdi   
;         addq    $8, top(%rip)      // top += 1

;         pushq   %rdi

; void** pop() { top -= 1; return (void**)(top[0]); }

;         movq    top(%rip), %rax    // rax = top
;         leaq    -8(%rax), %rdx     // rdx = rax - 1
;         movq    %rdx, top(%rip)    // top = rdx
;         movq    -8(%rax), %rax     // rax = rax[-1]

;         popq    %rax


; void push_result() { push(result); }

;         movq    top(%rip), %rax    // rax = top
;         movq    result(%rip), %rdx // rdx = result
;         movq    %rdx, (%rax)       // rax[0] = rdx
;         addq    $8, top(%rip)      // top = top + 1

;         pushq   result(%rip)

; call

;  popq    %rdx
;  pushq   env(%rip)
;  ... new environment based on %rdx and result
;  ... pointer to env = (body-addr, arg-value) is saved on stack
;  call    *(%rdx)
;  popq env(%rip)
;  ... pops the value saved earlier, specifically the env
