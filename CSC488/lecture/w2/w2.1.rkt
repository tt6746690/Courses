#lang racket

; recall the lambda calculus:
;  <identifier>
;  (λ (id) expr)
;  (e1 e2)
;     Evaluate e1.
;     Evaluate e2.

; examples...
#; x
#; (λ (x) x)
#; ((λ (x) x) x)
; 1. Evaluate (λ (x) x).
; 2. Evaluate x.
#; ((λ (x) x) (λ (y) y))
; 1. Evaluate (λ (x) x).
;    Remember it.
; 2. Evaluate (λ (y) y).
; Call the value from #1 with value from #2
; Remember which is the current environment
; Evaluate body of #1, i.e. x
;    In the environment where x is the value from #2
#; ((λ (x) (λ (y) (x y))) (λ (z) z))
; Evaluate (λ (x) (λ (y) (x y)))
; Remember the result
; Evaluate (λ (z) z)
; Pop to gget tthe function to call
; Push the current environment.
; Make new environment with x as λ0
; Evaluate (λ (y) (x y)) in that environment, to λ2
;   That's a closure: (λ (y) (x y)) along with x = λ1
; Pop the environment
#; (((λ (x) (λ (y) (x y))) (λ (z) z)) (λ (a) a))
; Evaluate fucntion expression to get λ2
; Push that
; Evaluate (λ (a) a) to get λ3
; Pop to get the function to call
; Push the current environment
; Make new environment with y = λ3
;   Attach this to λ2's environment
; Evaluate (x y) in that environment

(local [(define-syntax-rule (#%app e1 e2) (app e1 e2))]
  (λ (x) x))