#lang racket #| CSC 324 - 2017 Fall - Assignment 1 - Part 1 / 2 |#

#| Due Saturday October 28th at Noon.
   May be done with one other student in the course. |#

#| ★ Implement ‘eva’, an algebraic interpreter for an augmented Lambda Calculus.

 The syntactic language of Terms is the Lambda Calculus, augmented with numeric literals
  and a special constant ‘add1’ that when called with a numeric literal adds one to it.

   (λ (<identifier>) <term>)
     - represented by a list containing:
       • the symbol λ
       • a list containing an identifier
       • a term
   (add1 <a-term>)
     - represented by a list containing the symbol add1, and a term
   (<f-term> <a-term>)
     - represented by a list containing two terms
   <identifier>
     - represented by a symbol
   <literal>
     - represented by a number

 The semantics of function call is eager by-value, by algebraic substitution.

   (add1 <a-term>)
     1. Evaluate <a-term>, assume it produces a numeric literal.
     2. Add one to the value of <a-term>.

   (<f-term> <a-term>)
     1. Evaluate <f-term>, assume it produces a λ term: (λ (<id>) <body>) .
     2. Evaluate <a-term>, producing a value v.
     3. Substitute v into <body> by replacing <id> in <body> with v.
        Respect scope: if <body> contains a λ term whose parameter is also <id>,
         do not replace <id> anywhere in that λ term.
     4. Evaluate the transformed body.

   Any other term.
     The value of the term is itself. |#


(provide eva sub)

(module+ test
  (require rackunit)
  ; Your design and tests:

  ; eva
  (check-equal? (eva 1) 1)     ; literal 
  (check-equal? (eva 'a) 'a)   ; identifier
  (check-equal? (eva '(add1 1)) 2)  ; (f-term a-term)
  (check-equal? (eva '((λ (x) 3) 1)) 3) ; body is literal
  (check-equal? (eva '((λ (x) x) 1)) 1) ; body is identifier
  (check-equal? (eva '((λ (x) (add1 x)) 0)) 1)  ; body is (f-term a-term)
  ; parameter has to be evaluated first
  (check-equal? (eva '((λ (x) (add1 x)) (add1 1))) 3) 
  ; body is (f-name, a-term) where f-name has to be evaluated first
  (check-equal? (eva '((λ (x) ((λ (x) (add1 x)) x)) (add1 1))) 3)
  ; have to substitute f-name body 
  (check-equal? (eva '((λ (x) ((λ (y) (add1 x)) x)) 10)) 11)
 
  ; sub
  (check-equal? (sub 'x 1 2) 2)      ; literal
  (check-equal? (sub 'x 1 'x) 1)     ; identifier
  (check-equal? (sub 'x 123 '(add1 x)) '(add1 123))  ; (f-term a-term)
  (check-equal? (sub 'x 123 '(add1 (add1 x))) '(add1 (add1 123)))  ; body is (f-term a-term)
  (check-equal? (sub 'x 1 '((λ (x) x) x)) '((λ (x) x) 1))
  (check-equal? (sub 'x 123 '(λ (y) (add1 x))) '(λ (y) (add1 123)))
  (check-equal? (sub 'x 0 '(λ (y) ((λ (x) x) x))) '(λ (y) ((λ (x) x) 0)))
  (check-equal? (sub 'y 0 '((λ (x) x) y)) '((λ (x) x) 0))
  (check-equal? (sub 'x 0 '((λ (x) x) x)) '((λ (x) x) 0))
  (check-equal? (sub 'x 0 '(λ (y) ((λ (x) x) x))) '(λ (y) ((λ (x) x) 0)))

  (check-equal? (sub 'x 0 '(add1 ((λ (x) (add1 ((λ (x) x) x))) x)))
                '(add1 ((λ (x) (add1 ((λ (x) x) x))) 0))) 

  )



(define (eva term)
  (match term
    [`(add1 ,a-term) (+ a-term 1)]
    [`(λ (,id) ,body) term]
    [`(,f-term ,a-term)
     (local [(define f (eva f-term))
             (define v (eva a-term))
             (define id (first (second f)))
             (define body (third f))]
       (eva (sub id v body))
       )]
    [_ term]))


#| ★ Implement algebraic substitution.

 sub : symbol Term Term → Term
 Substitute value for id in term, respecting scope. |#

(define (sub id value term)
  (define (sub′ e) (sub id value e))
  (match term
    [`(λ (,id′) ,body)
     (list 'λ (list id′)
           (cond [(equal? id′ id) body]
                 [else (sub′ body)]))]
    [`(,f-term ,a-term)
     (list (sub′ f-term) (sub′ a-term))]
    [_ (cond [(equal? term id) value]
             [else term])]))
