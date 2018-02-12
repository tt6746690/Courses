#lang racket


#| Control Flow

  (e1 e2)
  evaluate e1
  evaluate e2
  call result of e1 on result of e2

  ((e1 e2) (e3 e4))
  evaluate e1
  push λ0
  evaluate e2
  call (λ0 e2)
  push λ1
  evaluate e3
  push λ2
  evaluate e4
  call (λ2 e4)
  call (λ1 e4)
|#

; escape continuation,
; escape from the expression
(+ 1
   (call/ec (λ (k)
              (k 5)
              2)))
; 6

(- (sqr (call/ec (λ (k) (sin 3)))))
; -0.01991485667481699
(- (sqr (call/ec (λ (k) (sin (k 3))))))
; -9

; call stack
;   -
;   sqr
;---------
;   sin




















