#lang racket

; Control flow and escaping continuations.

#;((+ 123) <e>)

; result, stack, env

; env: E0
; code for (+ 123)
; result: closure (+ 123)
; push result
; stack: closure (+ 123)

; <e> - can *extend* the stack temporarily
;     - can mess with the environment and result
;     - when done:
;         stack is where it was
;         result is the value of <e>
;         environment is restored
;         and we continue on to ★

; result : value of <e>

; ★ call
;   f = pop();
;   push(env);
;   env = new environment under f's environment with result as argument
;   stack: E0
;   push ★★
;   stack: E0
;          ★★
;   jump to body of closure (+ 123)
;     result = ...
;     jump to address on top of the stack
;   ★★
;   env = pop();
;   stack: -
;   env: E0
;   result: result of that body

#;((+ 123) <e>)
#;((+ 123) (call/ec (λ (k) <e>)))
; env: E0
; stack: closure (+ 123)

; stack: closure (+ 123)
;        closure call/ec
; code for (λ (k) <e>)
; result : closure (λ (k) <e>)
; call
;   stack: closure (+ 123)
;          E0
;          ★★★
;   jump to body of closure call/ec
;     make something that when called:
;       restores the stack pointer
;       sets result to its argument
;       then returns
;     pass that to the argument (λ (k) <e>)

(+ 123 (call/ec (λ (k) (+ 4000 (k 5000) (/ 1 0)))))
#;
(define (call/ec f)
  (define current-stack-pointer stack-pointer)
  (define k (λ (r)
              (set! stack-pointer current-stack-pointer)
              (set! result r)))
  (f k))

(define raise (λ (exception) (displayln 'uncaught)))

(define-syntax-rule (try handler
                         body ...)
  
  (call/ec (λ (k)
             (define chain raise)
             (set! raise (λ (exc)
                           (set! raise chain)
                           (k (handler exc))))
             (define result (let () body ...))
             (set! raise chain)
             result)))

(try (λ (exc) 0)
     (add1 (if (zero? (random 2)) 1 (raise 'hi))))