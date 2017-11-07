#lang racket

(require "mm.rkt")

; Let's make this
#;(class (Point a b)
    [a a]
    [b b]
    [size (sqrt (+ (sqr a) (sqr b)))]
    [double (set! a (* 2 a))
            (set! b (* 2 b))])
;  mean this:
#;(define (Point a b)
    (λ (message)
      (match message
        ['a a]
        ['b b]
        ['size (sqrt (+ (sqr a) (sqr b)))]
        ['double (set! a (* 2 a))
                 (set! b (* 2 b))])))

(define-syntax class
  (syntax-rules ()
    [(class (class-id init-id ...)
       [method-id body-expr
                  ...]
       ...)
     (define (class-id init-id ...)
       (λ (message)
         (match message
           ['method-id body-expr
                       ...]
           ...)))]))

(class (Point a b)
    [a a]
    [b b]
    [size (sqrt (+ (sqr a) (sqr b)))]
    [double (set! a (* 2 a))
            (set! b (* 2 b))])

(define p0 (Point 3 4))
(p0 'size)
(define p1 (Point 8 15))
(p0 'double)
(p0 'size)
(p1 'size)
