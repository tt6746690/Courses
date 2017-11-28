#lang racket #| Manipulating control flow without continuations. |#

#| Modelling control flow with the lambda calculus also requires more than small local changes.

 For a more natural, but still minimal, foundation for non-concurrent programming languages:
  add continuations, and expand the memory model to include runtime stacks.

 The standard transformation to express advanced control flow in the pure lambda calculus is
  a technique called “Continuation Passing Style” [CPS]. It's used in some compilers to unify
  the treatment of the specific control flow operations of a language. As a style it comes up,
  for example, in Javascript asynchronous programming.

 Pure functional programming has also identified and captured manipulation of control flow,
  and abstracted the commonalities with manipulating state. |#


; Let's express the problem of evaluating an expression involving binary divisions, and
;  handling the “early return” or “exception throwing” for division by zero.

; For simplicity, we'll use #false as the error result.

; Without return/throw/raise, nor explicit continuations, we check each result for #false,
;  short-circuiting if we encounter it, and propagate it up.
; You might be familiar with this style from CSC209.

(define (eval′ e)
  (cond [(number? e) e]
        [else (define r0 (eval′ (first e)))
              ; This is very bad style, but I want to make the #falses explicit:
              (cond [(equal? r0 #false) #false]
                    [else (define r1 (eval′ (third e)))
                          (cond [(equal? r1 #false) #false]
                                [else (cond [(zero? r1) #false]
                                            [else (/ r0 r1)])])])]))

(require rackunit)

(check-equal? (eval′ '((1 ÷ (5 ÷ 2)) ÷ (2 ÷ 3))) 3/5)
(check-equal? (eval′ '((1 ÷ (0 ÷ 2)) ÷ (2 ÷ 3))) #false)

; We can make a simple Do notation for that, to write code in the “maybe monad”:
(define-syntax Do
  (syntax-rules (←)
    [(Do (id ← e)
         clause ...
         r)
     (local [(define id e)]
       (cond [(equal? id #false) #false]   ; propagate #false up ...
             [else (Do clause
                       ...
                       r)]))]
    [(Do r)
     r]))

(define (eval e)
  (cond [(number? e) e]
        [else (Do (r0 ← (eval (first e)))
                  (r1 ← (eval (third e)))
                  (_  ← (not (zero? r1))) ; This acts as an assertion.
                  (/ r0 r1))]))

(check-equal? (eval '((1 ÷ (5 ÷ 2)) ÷ (2 ÷ 3))) 3/5)
(check-equal? (eval '((1 ÷ (0 ÷ 2)) ÷ (2 ÷ 3))) #false)