#lang typed/racket #| CSC324 2017 Fall Lab : Maybe Type |#

(require/typed typed/racket
               [+ (Real * → Real)]
               [- (Real Real * → Real)]               
               [* (Real * → Real)]
               [/ (Real Real * → Real)]
               [sqr (Real → Real)]
               [abs (Real → Real)])

(struct Success ([result : Real]) #:transparent)
(struct Failure () #:transparent)
(define-type Maybe (U Success Failure))

; “Lighter” syntax for unary and binary curried function definition with matching.
; See ÷ and 3+maybe below.
(define-syntax-rule (define1 id [pattern result] ...)
  (define/match (id x) [(pattern) result] ...))
(define-syntax-rule (define2 id [pattern1 pattern2 result] ...)
  (define/match ((id x) y) [(pattern1 pattern2) result] ...))
; note pattern1 is for x and pattern2 is for y

(: ÷ : Real → (Real → Maybe))
(define2 ÷
  [_ 0 (Failure)]
  [x y (Success (/ x y))])

((÷ 2) 3)
((÷ 2) 0)

(: 3+ : Real → Real)
(define (3+ r) (+ 3 r))

(: 3+maybe : Maybe → Maybe)
(define1 3+maybe
  [(Failure) (Failure)]
  [(Success x) (Success (3+ x))])

; example 
(3+maybe (Success 1))
(3+maybe (Failure))

#| ★ Define ‘lift’, to take a (Real → Real), and produce a (Maybe → Maybe) that behaves the same
      except propagates Failure. |#

(: lift : (Real → Real) → (Maybe → Maybe))
(define2 lift
  [f (Failure) (Failure)]
  [f (Success arg) (Success (f arg))])




(define reciprocal (÷ 1))
; tests
((lift sqr) ((lift 3+) (reciprocal 0)))
((lift sqr) ((lift 3+) (reciprocal 2)))
