#lang racket

(require "mm.rkt")

; Variadic functions.
#;(λ <identifier> ; Single identifier without parentheses.
    <body>
    ...)

((λ v ; Parameter gets bound to a list of the arguments.
   (+ (first v) (third v)))
 1 20 300 4000 50000)

; Like:
#;(local [(define v (list 1 20 300 4000 50000))]
    (+ (first v) (third v)))

; Let's make this
#;(class (Point a b)
    [(a) a]
    [(b) b]
    [(size) (sqrt (+ (sqr a) (sqr b)))]
    [(scale c) (set! a (* c a))
               (set! b (* c b))])
;  mean this:
#;(define (Point a b)
    (λ method+arguments
      (match method+arguments
        ['(a) a]
        ['(b) b]
        ['(size) (sqrt (+ (sqr a) (sqr b)))]
        [`(scale ,c) (set! a (* c a))
                     (set! b (* c b))])))

(define-syntax class
  (syntax-rules ()
    [(class (class-id init-id ...)
       [(method-id parameter-id ...) body-expr
                                     ...]
       ...)
     ; The memory modeller can take the keyword '#:name' in front of the function name
     ;  in a 'define', to tell it to use the function's name in the name it generates
     ;  for the corresponding λ(s).
     (define (#:name class-id init-id ...)
       ; The memory modeller can take '#:name <identifier>' after λ, to tell it to use
       ;  <identifier> in the name it generates for the corresponding λ(s).
       (λ #:name class-id
         message+arguments
         (match message+arguments
           [`(method-id ,parameter-id ...) body-expr
                                           ...]
           ...)))]))

(class (Point a b)
  [(a) a]
  [(b) b]
  [(size) (sqrt (+ (sqr a) (sqr b)))]
  [(scale c) (set! a (* c a))
             (set! b (* c b))])

(define p (Point 3 4))
(p 'size)
(p 'scale 10)
(p 'size)
(define p1 (Point 8 15))
(p1 'size)
(p1 'scale 3)
(p1 'size)
