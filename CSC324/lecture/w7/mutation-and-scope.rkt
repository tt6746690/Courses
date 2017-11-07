#lang racket

(require "mm.rkt")

; Create and initialize a variable.
(define x 0)
x
; Update the contents of the variable.
(set! x 1)
x

(define-syntax ++
  (syntax-rules ()
    [(++ x) (set! x (+ x 1))]))

(++ x) #;(set! x (+ x 1))

(define y 20)
(++ y) #;(set! y (+ y 1))
x
y

(define a 300)

(define (increment a)
  (set! a (+ a 1))
  a)

(increment x)
#;(increment 2)
; The body of increment runs in a new local scope with an ‘a’ shadowing the top-level ‘a’.
x ; Unchanged.


(increment a)
#;(increment 300)
a ; Unchanged.

