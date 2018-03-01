#lang racket #| Compile A2's Language L2 to Language X2 |#

#|
reference:
  https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf
|#

(provide L2→X2
         Mac? heap-size postamble)

(require "A2.L2.rkt")
(module+ test (require rackunit))

; Whether to emit code for the Mac, that Apple's gcc wrapper for clang handles. |#
(define Mac? (make-parameter #true))

; Size of the heap.
(define heap-size (make-parameter (/ (* 4 (expt 2 30)) 8))) ; 4G of 8-byte data.

; Code to append to main.
;
; If you put the resulting assembly code into a file file.s then the following postamble
;  prints the execution time and lowest byte of the result to the console if run with:
;   gcc -o file file.s ; time ./file ; echo $?
;
(define postamble (make-parameter "movq %rcx, %rax")) ; Return result.

#| X2
   ==
 Language X2 is a subset of 64-bit x86 assembly language, which we'll emit in the AT&T syntax.
 Details are in the rest of this file. |#

#| Machine Model
   =============
 Our current model of execution has a few global variables, which are frequently accessed and updated,
  and a stack with frequent stack-like operations. Many machine architectures provide the following
  model, and optimize for this pattern of use.

 The Model
 ---------
 Program: a sequence of statements.
 Execution: sequential, except after a statement that explicitly sets the current statement address.
 State: a fixed set of global variables, a stack, a large array, and a current statement address. |#

#| Global Variables
   ================
 The global variables in a CPU are called registers.

 From our point of view the general purpose X2 registers are all interchangeable. We'll use:

   register : use
   --------------
   a        : temporary variable (accumulator)
   c        : expression result  (counter)
   10       : next location to allocate in heap
   11       : current environment

 In 64-bit x86 with the AT&T syntax we refer to them as %rax, %rcx, %r10, and %r11.
 The names are not meaningful so let's hide them. |#

(define (register name) (~a '% name))
(define temp   (register 'rax))
(define result (register 'rcx))
(define next   (register 'r10))
(define env    (register 'r11))

#| Setting and Accessing Registers
   =============================== |#

(module+ test
  ; result = temp
  (check-equal? (movq temp result) "movq %rax, %rcx")
  ; result += temp
  (check-equal? (addq temp result) "addq %rax, %rcx"))

(define (movq from to) (~a 'movq " " from   ", " to))
(define (addq from to) (~a 'addq " " from   ", " to))

#| Integer Constants
   =================
 Integer constants are prefixed with "$".
 They can appear as ‘from’ in movq and addq. |#

(module+ test
  ; temp = 488
  (check-equal? (movq (constant 488) temp) "movq $488, %rax"))

(define (constant i) (~a '$ i))

#| Addresses of Statements
   =======================
 We can refer to the address of a statement by putting a label before the statement, and
  then use the label. In particular, we can change the execution order by jumping to a
  statement's address.

 We wont jump to [as opposed to call] stored locations, only explicit labels.

 To increase portability and flexibility, without much effort, we'll “mangle” labels by
  potentially adding an underscore [for the Mac's gcc wrapper around clang], and make them
  relative to the current instruction pointer [reasons and details aren't important for us]
  This does make them count as offset dereferences, and the limitation of the previous
  section applies. |#

; main()
;   temp = make_add
;   goto main
#;(labelled 'main
            (movq (label-reference 'make_add) temp)
            (jmp 'main))

(define (mangle name) (~a (if (Mac?) '_ "") name))
(define (labelled name . lines) (list (~a (mangle name) ':)
                                      lines))
(define (label-reference name) (~a (mangle name) "@GOTPCREL(%rip)"))

(define (jmp-label name) (~a 'jmp " " (mangle name)))

#| The Stack
   =========
 We can push a value [constant, or contents of a register], and pop a value into a register.

 Also, we can “call” a statement address [see “Addresses of Statements” below] that's stored
  in a register, which:
    1. Pushes the address of the statement that follows the call.
    2. Jumps to the address that's in the register.

 Coversely, we can “return”, which pops an address that's stored on the stack and jumps to it. |#

(define (pushq from) (~a 'pushq " "   from))
(define (popq  to)   (~a 'popq  " "   to))
(define (callq from) (~a 'call  " *(" from ")"))
(define (retq)       (~a 'retq))

#| Dereferencing and Pointer Arithmetic
   ====================================
 We'll store 64-bit data in our heap: the nth piece of data at an address is 8×n bytes after it.

 We can dereference a register containing an address, with an optional offset.
 Most ‘from’s or ‘to’s in the statements we're using can be a dereference, but not at the same time
  in a single statement. |#

(module+ test
  ; result = temp[0]
  (check-equal? (movq (★ temp) result) "movq 0(%rax), %rcx")
  ; result[488] = temp
  (check-equal? (movq temp (★ result 488)) "movq %rax, 3904(%rcx)"))


(define (⊕ offset) (* 8 offset))
(define (★ register [offset 0]) (~a (⊕ offset) "(" register ")"))

#| Conditional Execution
   =====================
 We can jump to an address conditionally, in particular on condition that two values are equal.
 Comparison sets a CPU flag, that various jump instructions react to.

 For comparison to a constant, the constant must be the first argument.

 We wont jump to calculated locations, only explicit labels. |#

; if (temp == result) goto main
#;(list (cmpq temp result)
        (je 'main))

(define (cmpq from-1 from-2) (~a 'cmpq " " from-1 ", " from-2))
(define (je-label name) (~a 'je " " (mangle name)))

#| L2 Statement to X2
   ==================
 Implement the seven functions needed to translate an L2 statement to an X2 statement or
  [possibly nested] list of statements.

 The nesting of the list structure is irrelevant: L2→X2 will flatten the results. |#

#|
set_result:
  movq %rdi, result(%rip)
push:
  movq top(%rip), %rax
  movq %rdi, (%rax)
  addq $8, top(%rip)
  ret
|#

(module+ test
  ; set_result
  (check-equal? (set_result 12) '("movq $12, %rcx"))
  ; push_result
  (check-equal? (push_result) '("pushq %rcx"))
  ; closure
  (check-equal? (closure 'make_add)
                '("movq _make_add@GOTPCREL(%rip), %rax"
                  "movq %rax, 0(%r10)"
                  "movq %r11, 8(%r10)"
                  "movq %r10, %rcx"
                  "addq $16, %r10"))
  ; call
  (check-equal? (call)
                '("popq %rax"
                  "pushq %r11"
                  "movq 8(%rax), %r11"
                  "movq %r11, 0(%r10)"
                  "movq %rcx, 8(%r10)"
                  "movq %r10, %r11"
                  "addq $16, %r10"
                  "call *(%rax)"
                  "popq %r11"))
  ; variable
  (check-equal? (variable 0)
                '("movq %r11, %rax" "movq 8(%rax), %rcx"))
  (check-equal? (variable 3)
                '("movq %r11, %rax" "movq 0(%rax), %rax"
                                    "movq 0(%rax), %rax"
                                    "movq 0(%rax), %rax" "movq 8(%rax), %rcx"))
  ; set
  (check-equal? (set 0)
                '("movq %r11, %rax" "movq %rcx, 8(%rax)"))
  (check-equal? (set 3)
                '("movq %r11, %rax" "movq 0(%rax), %rax"
                                    "movq 0(%rax), %rax"
                                    "movq 0(%rax), %rax" "movq %rcx, 8(%rax)"))
  ; label, jump, jump_false
  (check-equal? (label 'main) '(("_main:" ())))
  (check-equal? (jump 'main) '("jmp _main"))
  (check-equal? (jump_false 'main) '("cmpq $0, %rcx" "je _main"))
  )
  
(define (l2→X2 l2) (match l2
                     [`(L2: set_result ,<i>) (set_result <i>)]
                     [`(L2: push_result) (push_result)]
                     [`(L2: closure ,<name>) (closure <name>)]
                     [`(L2: call) (call)]
                     [`(L2: variable ,<n>) (variable <n>)]
                     [`(L2: set ,<n>) (set <n>)]
                     [`(L2: label ,<name>) (label <name>)]
                     [`(L2: jump ,<name>) (jump <name>)]
                     [`(L2: jump_false ,<name>) (jump_false <name>)]))

; Set result to integer i.
(define (set_result i)
  `(,(movq (constant i) result)))

; Push result onto the stack.
(define (push_result)
  `(,(pushq result)))

; Put a closure on the heap.
;   A closure is a pair of body address and an env.
;   The closure is put at the address referred to by next, and then next is adjusted
;    to point to the next place to put a pair.
(define (closure name)
  `(,(movq (label-reference name) temp)
    ,(movq temp (★ next))
    ,(movq env (★ next 1))
    ,(movq next result)    ; save pointer to closure to result, not sure if needed as of now
    ,(addq (constant 16) next)))



; Call the closure that's on the stack, with the argument that's in result.
;   Temporarily stores env on the stack.
;   Sets env to a new environment containing the closure's environment and the argument.
;   Calls the closure.
(define (call)
  `(
    ; void** f = pop(); push(env);
    ,(popq temp)  
    ,(pushq env)
    ; env = environment(f[1], result);
    ,(movq (★ temp 1) env)  ; env = closure-env
    ,(movq env (★ next))    ; putting (env, result) on heap
    ,(movq result (★ next 1))
    ,(movq next env)        ; env = ptr to newly created env
    ,(addq (constant 16) next)
    ; ((void(*)())(f[0]))();
    ,(callq temp)
    ; env = pop();
    ,(popq env)))


; Puts the value of the variable n levels up from env, into result.
;   To “loop” n times: emits n statements.
(define (variable n)
  (append `(,(movq env temp))
          (for/list [(i n)] (movq (★ temp) temp))
          `(,(movq (★ temp 1) result))))

; Sets the variable n levels up from env, to the value of result.
;   To “loop” n times: emits n statements.
(define (set n)
  (append `(,(movq env temp))
          (for/list [(i n)] (movq (★ temp) temp))
          `(,(movq result (★ temp 1)))))

; Names the current statement address.
(define (label name)
  `(,(labelled name)))

; Jumps to a named statement address.
(define (jump name)
  `(,(jmp-label name)))

; Jumps to a named statement address, if result is false.
;   False is represented by 0.
(define (jump_false name)
  `(,(cmpq (constant 0) result)
    ,(je-label name)))

#| L2 to X2
   ======== |#

(define (L2→X2 compiled)
  (match-define (compiled:L2 code λs) compiled)
  (map (curryr ~a "\n")  ; same as curry, but args collected in opposite direction
       (flatten (list (~a '.globl "  " (mangle 'main))
                      RTL
                      (map λ→X2 λs)
                      (labelled 'main
                                (movq (label-reference 'heap) next)
                                (map l2→X2 code)
                                (postamble)
                                (retq))
                      (~a '.comm  "  " (mangle 'heap) "," (heap-size) "," (if (Mac?) 4 32))))))

; For a compiled λ from L2: the code for its body, including a return, labelled by the name of the λ.
(define (λ→X2 a-λ) (labelled (first a-λ)
                             (map l2→X2 (second a-λ))
                             (retq)))


#| Runtime Library
   =============== |#

; Addition and Multiplication
; ---------------------------

; Roughly, we've been treating addition as if it's:
#;(define + (λ_make_add (variable_1)
                        (λ_add (variable_0)
                               (primitive-addition variable_0 variable_1))))

; L1→L2 translates ‘+’ to a statement that creates a make_add closure.
(module+ test
  (check-equal? (L1→L2 '(L1: var +)) (compiled:L2
                                      '((L2: closure make_add))
                                      '())))

#|
void add() {
  variable(1);
  long long temp = (long long)result;
  variable(0);
  result = (void*)(temp + (long long)result); }
void make_add() { closure(add); }

|#

; Put X2 versions of make_add and add in RTL below.
; Similarly, find the 64-bit x86 instruction for multiplication, and add multiplication.

(module+ test
  (check-equal? (flatten add)
                '("_add:"
                  "movq %r11, %rax"
                  "movq 0(%rax), %rax"
                  "movq 8(%rax), %rcx"
                  "pushq %rcx"
                  "movq %r11, %rax"
                  "movq 8(%rax), %rcx"
                  "popq %rax"
                  "addq %rax, %rcx"
                  "retq"))
  (check-equal? (flatten make_add)
                '("_make_add:"
                  "movq _add@GOTPCREL(%rip), %rax"
                  "movq %rax, 0(%r10)"
                  "movq %r11, 8(%r10)"
                  "movq %r10, %rcx"
                  "addq $16, %r10"
                  "retq"))
  )


(define add
  (labelled 'add
            (variable 1)
            (pushq result)
            (variable 0)
            (popq temp)
            (addq temp result)
            (retq)))

(define make_add
  (labelled 'make_add
            (closure 'add)
            (retq)))



; Escape Continuations
; --------------------

; The continuation of an expression is:
;
;   The state of the heap, and the stack and env before the expression begins evaluating,
;    and the address of the statement after the expression's statements, with that statement
;    waiting to work with the result.

; Write out the compilation of (call/ec f) to convince yourself that the continuation of
;  that expression is on the stack. And convince yourself that setting the result to v
;  and executing a return with the stack in that state continues as if the value of
;  (call/ec f) is v [and any side-effects until then are in the heap and persist].

; (call/ec f) calls f with an escape continuation k, where (k v) escapes the evaluation
;  of (call/ec f) to produce v. Roughly, we treat call/ec as:
#;(λ_call_ec (f)
             (f ((λ_make_ec (saved-stack-pointer)
                            (λ_ec (result) (set! stack-pointer saved-stack-pointer)
                                  result))
                 stack-pointer)))

; The CPU's stack pointer is a register:
(define stack-pointer (register 'rsp))


(define ec
  (labelled 'ec
            (variable 1)
            (movq result stack-pointer)
            (variable 0)
            (retq)))

(define make_ec
  (labelled 'make_ec
            (closure 'ec)
            (retq)))

(define call_ec
  (labelled 'call_ec
            (variable 0)
            (push_result)         ; stack: [f]
            (closure 'make_ec)
            (push_result)         ; stack: [f make_ec]
            (movq stack-pointer result)
            (call)                ; (make_ec stack-pointer)   now stack: [f]
            (call)                ; (f result)
            (retq)))



#;(call/ec f)
; Call unary function f, passing in the continuation k of the (call/ec f) expression.
; Calling k with argument v during the execution of (call/ec f) aborts that expression,
;  and uses v as its result value.

#;(+ 1 (call/ec (λ (k) 100)))  ; output 101
(define callec-testcase '(L1: app (L1: app (L1: var +) (L1: datum 1))
                              (L1: app (L1: var call/ec)
                                   (L1: λ 0 (L1: datum 100)))))

#;(+ 1 (call/ec (λ (k) (k 10))))  ; output 11
(define callec2-testcase '(L1: app (L1: app (L1: var +) (L1: datum 1))
                               (L1: app (L1: var call/ec)
                                    (L1: λ 0 (L1: app (L1: var 0) (L1: datum 10))))))

#;(- (sqr (call/ec (λ (k) (sin (k 3))))))

; call stack:
;   -
;   sqr
; -------- k
;   sin
; Calling k resets control flow to that line, throwing away the waiting call to sin.




; A2's L1→L2 translates ‘call/ec’ to a statement that creates a call_ec closure.
(module+ test
  (check-equal? (L1→L2 '(L1: var call/ec)) (compiled:L2
                                            '((L2: closure call_ec))
                                            '())))

; Put X2 versions of calss.

; Roughly, we've been treating “less than” as if it's:
#;(define < (λ_make_less_than (variable_1)
                              (λ_less_than (variable_0)
                                           (primitive-less-than variable_1 variable_0))))

; L1→L2 translates ‘<’ to a statement that creates a make_less_than closure.
(module+ test
  (check-equal? (L1→L2 '(L1: var <)) (compiled:L2
                                      '((L2: closure make_less_than))
                                      '())))

; The CPU flags set by a comparison can be stored as a byte, which we then “widen” to a 64 bit value.
; if result < temp
;    result = 1
; else
;    result = 0
#;(list (cmpq temp result)
        (setl result-byte)
        (movzbq result-byte result))

(define (setl to) (~a 'setb  " " to))
(define result-byte (register 'cl))
(define (movzbq from-1 from-2) (~a 'movzbq " " from-1 ", " from-2))

; Put X2 versions of make_less_than and less_than in RTL below.


(define RTL
  `(,add ,make_add ,call_ec ,make_ec ,ec))

; testing stuff
(define (L1→X2 x) (L2→X2 (L1→L2 x)))

(define app-testcase '(L1: app (L1: λ 0 (L1: var 0)) (L1: datum 200)))
(define add-testcase '(L1: app (L1: app (L1: var +) (L1: datum 12)) (L1: datum 24)))


(define out (open-output-file "file.s" 	#:exists 'replace))
(define assembly (L1→X2 callec2-testcase))
(map (λ (x) (display x out)) assembly)
(close-output-port out)

