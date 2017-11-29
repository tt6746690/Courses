#lang racket #| CSC324 2017 Fall Assignment 2 : Due Wednesday November 29 at 6PM. |#

#| The Maybe Monad.

 In this assignment you implement the functional API for computation that propagates false as failure,
  and use that to implement the associated Do notation. |#

(provide >> >>=
         ÷ √ ln
         E1 Do E2 E3)

(module+ test (require rackunit)
  
  ; Implement ‘>>’, called “then”, that takes two *expressions* and produces an expression that:
  ;   1. Evaluates the first expression.
  ;   2. If that produces false then that is the result.
  ;   3. Otherwise, evaluates the second expression and that is the result.
  
  (check-false (>> (not (zero? 0)) (/ 324 0)))
  (check-equal? (>> (not (zero? 324)) (/ 324 324))
                  1)
  (check-false (>> (number? "324") (>> (not (zero? "324")) (/ 324 "324"))))

  ; Implement functions ÷, √, and ln, that produce false if dividing by zero, taking the square root
  ;  of a negative number, or taking the logarithm of a non-positive number.
  ; Use ‘>>’ appropriately in the implementations.
  ; Implement ÷ curried: taking a number, and returning a unary function ready to divide a number.
  (check-false (√ -1))
  (check-equal? (√ 324) 18)
  (check-false ((÷ 1) 0))
  (check-equal? ((÷ 324) 18) 18)
  (check-false (ln 0))
  (check-equal? (ln 324) (log 324))

  
  ; Implement *function* ‘>>=’, called “bind”, that takes two arguments and:
  ;  1. If the first argument is false then that is the result.
  ;  2. Otherwise, calls the second argument on the first.
  ; Use ‘>>’ appropriately in the implementation.
  (check-false (>>= -1 √))
  (check-false (>>= (>>= -1 √) ln))
  (check-equal? (>>= (>>= (>>= 324 √) (÷ 1)) ln)
                (log (/ (sqrt 324)))))

(define-syntax-rule (>> e1 e2)
  (if (not e1) e1 e2))

(define (÷ numerator)
  (λ (denominator) (>> (not (zero? denominator)) (/ numerator denominator))))

; note function negative? expect a real?,
; so if a-non-negative-number is #f, then invalid argument,
; so have to use >>= in this case to propagate #f
(define (√ a-non-negative-number)  
  (>> (not (negative? a-non-negative-number)) (sqrt a-non-negative-number)))

(define (ln a-positive-number)
  (>> (> a-positive-number 0) (log a-positive-number)))

(define (>>= arg f)
  (>> (not (false? arg)) (f arg)))


; Consider this language of arithmetic expressions:
;   <numeric-literal>
;      - represented by a number
;   (√ <ae>)
;      - represented by a list with the symbol √  and an arithemtic expression
;   (ln <ae>)
;      - represented by a list with the symbol ln and an arithemtic expression
;   (<ae> ÷ <ae>)
;      - represented by a list with an arithmetic expression, symbol ÷, and arithemtic expression
  
; Implement function ‘E1’ to evaluate such expressions, producing false if any of the computations
;  are invalid according to the earlier restrictions for square root, logarithm, and division.
; Use pattern matching appropriately, along with ‘>>’ and ‘>>=’ for propagating false.
; In particular, do not use any other conditionals, nor boolean operations or literals.
; Use quasiquotation as appropriate for the patterns, but nothing else from match's pattern language
; [e.g. don't use ‘?’, nor #:when].

(module+ test (require rackunit)
  ; check literal
  (check-equal? (E1 1) 1)

  ; check non-nested return either false or result of operation properly 
  (check-false (E1 '(√ -1)))
  (check-equal? (E1 '(√ 324)) 18)
  (check-false (E1 '(1 ÷ 0)))
  (check-equal? (E1 '(324 ÷ 18)) 18)
  (check-false (E1 '(ln 0)))
  (check-equal? (E1 '(ln 324)) (log 324))

  ; nested, value OK
  (check-equal? (E1 '(√ (√ 81))) 3)
  (check-equal? (E1 '((√ 4) ÷ (4 ÷ 2))) 1)
  (check-equal? (E1 '(ln (1 ÷ (√ 324)))) (log (/ (sqrt 324))))

  ; nested, value false should be propagated
  (check-false (E1 '(ln (√ -1))))

  ; some additional tests
  (check-equal? (E1 '(√ 9)) 3)
  (check-equal? (E1 '(√ -1)) #f)
  (check-equal? (E1 '(√ (√ 81))) 3)
  (check-equal? (E1 '(√ (√ -1))) #f)
  (check-equal? (E1 '(√ (ln 81))) (sqrt (ln 81)))
  (check-equal? (E1 '(ln (ln 0))) #f)
  (check-equal? (E1 '(10 ÷ 5)) 2)
  (check-equal? (E1 '(90 ÷ (√ 81))) 10)
  (check-equal? (E1 '(90 ÷ 0)) #f)
  (check-equal? (E1 '((ln -1) ÷ 0)) #f)
  )


(define (E1 expr)
  (match expr
    [`(,<ae1> ÷ ,<ae2>) (>>= (E1 <ae2>) (÷ (E1 <ae1>)))]
    [`(√ ,<ae>)  (>>= (E1 <ae>) √)]
    [`(ln ,<ae>) (>>= (E1 <ae>) ln)]
    [x  x]
    ))

; Implement ‘Do’, using ‘>>’ and ‘>>=’ appropriately.
;
; It takes a sequence of clauses to be evaluated in order, short-circuiting to produce false if any
;  of the clauses produces false, producing the value of the last clause.
;
; Except for the last clause, a clause can be of the form
#;(identifier ← expression)
;  in which case its meaning is: evaluate the expression, and make the identifier refer to the
;  value in subsequent clauses.
;
; Don't use any local naming [local, let, match, define, etc] except for λ parameters:
;  recall that ‘let’ is just a notation for a particular lambda calculus “design pattern”.
  
(module+ test
  (check-equal? (Do 324)
                324)
  (check-false (Do #false
                   (/ 1 0)))
  (check-false (Do (r1 ← (√ -1))
                   (r2 ← (ln (+ 1 r1)))
                   ((÷ r1) r2)))
  (check-false (Do (r1 ← (√ -1))
                   (r2 ← (ln (+ 1 r1)))
                   ((÷ r1) r2)))

  ; some additional tests
  (check-equal? (Do 324)
                324)
  (check-false (Do #false
                   (/ 1 0)))
  (check-false (Do #true #false
                   (/ 1 0)))
  (check-false (Do #true #true (√ -1) #false
                   (/ 1 0)))
  (check-false (Do #true #true  #false
                   (/ 1 0)))
  (check-false (Do (r1 ← (√ -1))
                   (r2 ← (ln (+ 1 r1)))
                   ((÷ r1) r2)))
  (check-false (Do (r1 ← (√ -1))
                   (r2 ← (ln (+ 1 r1)))
                   ((÷ r1) r2)))
  )

(define-syntax Do
  (syntax-rules (←)
    [(Do (id ← e)
         clause ...
         r)
     (>>= e (λ (id) (Do clause ... r)))]
    [(Do e
         clause ...
         r)
     (>>= e (λ (_) (Do clause ... r)))]
    [(Do r)
     r]))

; Implement ‘E2’, behaving the same way as ‘E1’, but using ‘Do’ notation instead of ‘>>’ and ‘>>=’.

(module+ test (require rackunit)
  ; check literal
  (check-equal? (E2 1) 1)

  ; check non-nested return either false or result of operation properly 
  (check-false (E2 '(√ -1)))
  (check-equal? (E2 '(√ 324)) 18)
  (check-false (E2 '(1 ÷ 0)))
  (check-equal? (E2 '(324 ÷ 18)) 18)
  (check-false (E2 '(ln 0)))
  (check-equal? (E2 '(ln 324)) (log 324))

  ; nested, value OK
  (check-equal? (E2 '(√ (√ 81))) 3)
  (check-equal? (E2 '((√ 4) ÷ (4 ÷ 2))) 1)
  (check-equal? (E2 '(ln (1 ÷ (√ 324)))) (log (/ (sqrt 324))))

  ; nested, value false should be propagated
  (check-false (E2 '(ln (√ -1))))

  ; some additional tests
  (check-equal? (E2 '(√ 9)) 3)
  (check-equal? (E2 '(√ -1)) #f)
  (check-equal? (E2 '(√ (√ 81))) 3)
  (check-equal? (E2 '(√ (√ -1))) #f)
  (check-equal? (E2 '(√ (ln 81))) (sqrt (ln 81)))
  (check-equal? (E2 '(ln (ln 0))) #f)
  (check-equal? (E2 '(10 ÷ 5)) 2)
  (check-equal? (E2 '(90 ÷ (√ 81))) 10)
  (check-equal? (E2 '(90 ÷ 0)) #f)
  (check-equal? (E2 '((ln -1) ÷ 0)) #f)
  )

(define (E2 expr)
  (match expr
    [`(,<ae1> ÷ ,<ae2>)
     (Do (x ← (E2 <ae1>))
         (y ← (E2 <ae2>))
         ((÷ x) y))]
    [`(√ ,<ae>)
     (Do (x ← (E2 <ae>))
         (√ x))]
    [`(ln ,<ae>)
     (Do (x ← (E2 <ae>))
         (ln x))]
    [x  x]
    ))

; Implement ‘E3’, behaving the same way as ‘E2’, by expanding each use of ‘Do’ notation in ‘E2’,
;  and also replacing ‘E2’ with ‘E3’. The result will be similar to your ‘E1’, but likely a bit
;  less elegant.

(module+ test (require rackunit)
  ; check literal
  (check-equal? (E3 1) 1)

  ; check non-nested return either false or result of operation properly 
  (check-false (E3 '(√ -1)))
  (check-equal? (E3 '(√ 324)) 18)
  (check-false (E3 '(1 ÷ 0)))
  (check-equal? (E3 '(324 ÷ 18)) 18)
  (check-false (E3 '(ln 0)))
  (check-equal? (E3 '(ln 324)) (log 324))

  ; nested, value OK
  (check-equal? (E3 '(√ (√ 81))) 3)
  (check-equal? (E3 '((√ 4) ÷ (4 ÷ 2))) 1)
  (check-equal? (E3 '(ln (1 ÷ (√ 324)))) (log (/ (sqrt 324))))

  ; nested, value false should be propagated
  (check-false (E3 '(ln (√ -1))))
  
  ; some additional tests
  (check-equal? (E3 '(√ 9)) 3)
  (check-equal? (E3 '(√ -1)) #f)
  (check-equal? (E3 '(√ (√ 81))) 3)
  (check-equal? (E3 '(√ (√ -1))) #f)
  (check-equal? (E3 '(√ (ln 81))) (sqrt (ln 81)))
  (check-equal? (E3 '(ln (ln 0))) #f)
  (check-equal? (E3 '(10 ÷ 5)) 2)
  (check-equal? (E3 '(90 ÷ (√ 81))) 10)
  (check-equal? (E3 '(90 ÷ 0)) #f)
  (check-equal? (E3 '((ln -1) ÷ 0)) #f)
  )

(define (E3 expr)
  (match expr
    [`(,<ae1> ÷ ,<ae2>)
     (>>= (E3 <ae1>)
          (λ (x) (>>= (E3 <ae2>)
                      (λ (y) ((÷ x) y)))))]
    [`(√ ,<ae>)
     (>>= (E3 <ae>)
          (λ (x) (√ x)))]
    [`(ln ,<ae>)
     (>>= (E3 <ae>)
          (λ (x) (ln x)))]
    [x  x]
    ))

