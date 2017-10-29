#lang racket #| CSC 324 - 2017 Fall - Exercise 5 - Part 2 of 2 |#

; Save this file in a directory containing the memory model library from lecture.
(require "mm.rkt")

; Run this to see the memory model trace of evaluation.
((λ (One)
   ((λ (Add1)
      (Add1 (Add1 One)))
    (λ (f)
      (λ (g)
        (λ (h) (g ((f g) h)))))))
 (λ (h) h))

(provide closures)

; ★ Fill in the rest of the table of closures.
; A closure is represented by a list with the symbol 'closure:, a λ term, and an environment.
; An environment contains variables and their values, from local to global. The values in an
;  environment are "references" to closures.
; Ignore the part of the top-level environment containing the variable ‘closures’.
(define closures
  '((λ5 (closure: (λ (g) (λ (h) (g ((f g) h))))
                  ((f λ4) (Add1 λ3) (One λ1))))
    (λ4 (closure: (λ (g) (λ (h) (g ((f g) h))))
                  ((f λ1) (Add1 λ3) (One λ1))))
    
    (λ3 (closure: (λ (f) (λ (g) (λ (h) (g ((f g) h)))))
                  ((One λ1))))
    (λ2 (closure: (λ (Add1) (Add1 (Add1 One)))
                  ((One λ1))))
    (λ1 (closure: (λ (h) h)
                  ()))
    (λ0 (closure: (λ (One)
                    ((λ (Add1)
                       (Add1 (Add1 One)))
                     (λ (f) (λ (g) (λ (h) (g ((f g) h)))))))
                  ()))))
