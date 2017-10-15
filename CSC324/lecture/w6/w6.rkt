#lang racket 
#|
    W6-hour1
|#

; #%app inserted into front of every funtion call
;     functions as to convert argument expression to values and invoke functions
;     semantics: eager and by value 
; define, quote, etc are not function calls 


; If want lazy evlauation
; wrap expression in a function to delay evaluation
; so now If is a notation not a function
(define-syntax If
  (syntax-rules ()
    [(If c t e) (c (Lambda () t) (Lambda () s))]))

#;(define (If c t e)
    (If c
        (t)
        (e)))

; lazy
#;(If (zero? (random 2))
      (λ () (/ 1 0))
      (λ () "ok"))



; motivation for mutation
; memory is limited, have to change values in limited space
; functional have no mutation but may require more space


#|
    W6-hour2 Mutation
    mutation has to be in core of language
|#

#{
  (define x 0)
  x ; 0
  (set! x 1)
  x ; 1



  #; (define (++ v) (set! v (+ v 1)))
  (define-syntax-rule (++ x)
    (set! x (+ x 1)))

  (++ x)   #; (set! x (+ x 1))
  x        #; 1}
; function call sets up new function frame (i.e. enviornment)

; Memory model is used to track mutation


(define (Point a b)
  (λ (message)
    (cond [(equal? message 'a) a]           ; methods... (but initially just message to a hash table)
          [(equal? message 'b) b]
          [((equal? message 'size)
            (sqrt (+ (sqr a) (sqr b))))])))

; closure
; algebraic model: substitute value of variable to scope (body) of lambda,
;                  the code itself codes the values
; memory model   : code grabs env when it is created...
;                  

; class Point
(define p (Point 3 4))   ; but p is a lambda
(p 'a)
(p 'b)

(define a 678)
(define p1 (Point 9 10))



#; (class (C id ...)
     (method-name body)
     ...)
#; (define (C id ...)
     (λ (message)
       (cond [(equal? message 'method-name) body]
             ...)))

; syntax for making classes
(define-syntax rule
  (class (C id ...)
     (method-name body)
     ...)
  (define (C id ...)
     (λ (message)
       (cond [(equal? message 'method-name) body]
             ...))))
; ... pattern matching

(class (3D-point x y z)
  (x x)
  (y y)
  (z z)
  (lenght (sqrt (+ (sqr x) (sqr y) (sqr z)))))





















