#lang racket

(define K (void))

#;(call/ec f)
; Call unary function f, passing in the continuation k of the (call/ec f) expression.
; Calling k with argument v during the execution of (call/ec f) aborts that expression,
;  and uses v as its result value.

(+ 1 (call/ec (λ (k) (+ 20 (k 300) (/ 1 0)))))

(+ 1 (call/ec (λ (k)
                ; Save the continuation.
                (set! K k)
                ; Abort with 20.
                (k 20)
                300)))

; Error: can only be used to escape, not to re-enter.
#;(K 0)

(- (sqr (call/ec (λ (k) (sin (k 3))))))

; call stack:
;   -
;   sqr
; -------- k
;   sin
; Calling k resets control flow to that line, throwing away the waiting call to sin.
