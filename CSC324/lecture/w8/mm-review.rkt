#lang racket

(require "mm.rkt")
(wait! #false)
(scale! 20)


(define x 1) ; Can't just algebraically substitute 1 for all references to this variable x.

(λ () 0)

(λ () x) ; The closure is this code along with the pointer to the current environment.
((λ () x)) ; The pointer is needed to look up x.

((λ (y) (+ x y)) 3)  ; 4
((λ (y) (+ 1 y)) 3)  ; 4
(set! x 6)
; Different than if we had substituted for x when we defined x.
((λ (y) (+ x y)) 3)  ; 9

(define f (λ (y) x)) ; f name in toplevel env, with a lambda def pointing to it
(f 4)  ; 6
(f 5)  ; 6

((λ () (λ () 2)))  ; #<procedure:λ7.>  the nested inner function

; Looks up the chain of environments for an x.
(((λ () (λ () x)))) ; 6

(((λ (x y) ; Shadows top-level x.
    (λ (x) ; Shadows that x.
      (+ x y)))  ; x = 13, y = 7
  11 7)
 13)
; 20

(define change 0)      ; toplevel env
(define make-constant  ; toplevel env
  (λ (y)
    (set! change (λ () (set! y 10))) 
    (λ () y)))
(define c8 (make-constant 8)) ; supplies y:8 as arg to make-constant,
                              ; 2 new lambdas λ13 and λ14 points to the y:8 box
; Seems equivalent to returning (λ () 8).
(c8)  ; 8
      ; (λ () 8) is evaluated 
(change) ; changes c9's closure's environment
         ; (set! y 10) is evaluated
         ; note the argument y:8 is mutated
(c8) ; But it isn't.
     ; 10
(define c9 (make-constant 9)) ; changes change as a side-effect
(c9)
(change) ; changes c9's closure's environment.
(c9) ; Produces 10.

#| Properties the model always has.
 1. Levels alternate: environment / closure / environment / closure / ...
 2. The parent-child relationship of closures in the tree of code matches the relationship
     in memory. |#
#;(λ1 (id1 ...)
      _
      ; Every instance of λ1.1 is under an [(id1 value) ...] environment under λ1
      (λ1.1 (id1.1 ...)
            _
            ; Every instance of λ1.1.1 is under an [(id1.1 value) ...] environment under λ1.1
            (λ1.1.1 (id1.1.1 ...) 
                    -
                    _)
            _
            ; Every instance of λ1.1.2 is under an [(id1.1 value) ...] environment under λ1.1
            (λ1.1.2 (id1.1.2 ...)
                    -
                    _))
      _
      ; Every instance of λ1.2 is under an [(id1 value) ...] environment under λ1
      (λ1.2 (id1.2 ...)
            _
            (λ1.2.1 (id1.2.1 ...)
                    -
                    _)
            _
            (λ1.2.2 (id1.2.2 ...)
                    -
                    _)))
#| 3. In an environment [(id value) ...], the values are *never* variables.
   4. Closure boxes *never* change. |#