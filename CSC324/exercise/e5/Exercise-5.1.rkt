#lang racket #| CSC 324 - 2017 Fall - Exercise 5 - Part 1 of 2 |#

(require rackunit)

; Consider this term in the Lambda Calculus augmented with add1 and numeric literals:
(check-equal? ((((λ (f) (λ (g) (λ (h) (g ((f g) h)))))
                 (λ (h) h))
                add1)
               0)
              2)

; In terms of our work building arithmetic in lecture, it corresponds to:
#;(((Add1 One) add1) 0)

; Add elements to the list ‘steps’: each term in the algebraic simplification to get to 2.
(define steps
  (list '((((λ (f) (λ (g) (λ (h) (g ((f g) h)))))
            (λ (h) h))
           add1)
          0)

        '(
          (
           (λ (g) (λ (h) (g (((λ (h) h) g) h))))
           add1)
          0)

        '(
           (λ (h) (add1 (((λ (h) h) add1) h)))
          0)

        '(add1 (((λ (h) h) add1) 0))

        '(add1 (add1 0))

        '(add1 1)

        ; ★ Add more steps here.
        2))

(provide steps)

(check-equal? (first steps) '((((λ (f) (λ (g) (λ (h) (g ((f g) h)))))
                                (λ (h) h))
                               add1)
                              0))
(check-equal? (length steps) 7)
(check-equal? (map list? steps) (list #true #true #true #true #true #true #false))
(check-equal? (last steps) 2)
