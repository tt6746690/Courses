#lang racket

; lets look at non-unary function calls

(require "tree.rkt") (current-font-size 20)

'((λ (x) (λ (y) (⊕ x y))
    30)
  40)

''((λ1 (x) (λ0 (y) (⊕ x y))
       30)
   40)
 

; closure λ1 = Ca
; push Ca
; result = 30
; call
;   Ea = environment under Ca's environment
;        with x = 30
;   push ·
;     do λ1 body while environment is Ea
;     closure λ0 = Cb
;   pop ·
; push Cb
; result = 40
; call
;   Eb = environment under Cb's environment
;        with y = 40
;   push ·
;     do λ0 body while envorinment is Eb
;     get 70
;   pop ·

(tree '(· ((Ca (λ1 ·)))
          ((Ea (· (x 30))) ((C0 (λ0 Ea)))
                           ((Eb (Ea (y 40)))))))

'((λ2 (x y) (⊕ x y))
  30
  40)

; closure λ2 = Cc
; push Cc
; ---  [stack: _ Cc]
; result = 30
; push result
; --- [stack: 30 Cc]
; result = 40
; --- 
; call-2
;   Ec = [? ? ?]
;   arg-0 = pop = 30 [? 30 ?]
;   pop = Cc
;   Ec = environment under CC's environment
;        with arg-0, arg-1 = result ; [· 30 40]
;   [stack: -]
;   push · [stack: ·]
;     do λ2 body while environment is Ec
;     get 70
;   pop · [stack: -]


(tree '(· ((Cc (λ1 ·)))
          ((Ec (· (x 30 y 40))))))



; ---
; closure λ1 = Ca
; push Ca [stack: Ca]
; result = 30
; call
;   pop = Ca [stack: -]
;   Ea = environment under Ca's environment
;        with x = 30
;   push · [stack: ·]
;     do λ1 body while environment is Ea
;     closure λ0 = Cb
;   pop = · [stack: -]
; ---
; push Cb
; result = 40
; call
;   Eb = environment under Cb's environment
;        with y = 40
;   push ·
;     do λ0 body while envorinment is Eb
;     get 70
;   pop ·

