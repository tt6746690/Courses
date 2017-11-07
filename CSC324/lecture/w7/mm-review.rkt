#lang racket

(require "mm.rkt")
;(wait! #false)
; (scale! 20)

; (define x 1) ; Can't just algebraically substitute 1 for all references to this variable x.
; 
; (λ () 0)
; 
; (λ () x) ; The closure is this code along with the pointer to the current environment.
; ((λ () x)) ; The pointer is needed to look up x.
; 
; ((λ (y) (+ x y)) 3)
; ((λ (y) (+ 1 y)) 3)
; (set! x 6)
; ; Different than if we had substituted for x when we defined x.
; ((λ (y) (+ x y)) 3)
; 
; (define f (λ (y) x))
; (f 4)
; (f 5)
; 
; ((λ () (λ () 2)))
; 
; ; Looks up the chain of environments for an x.
; (((λ () (λ () x))))
; 
; (((λ (x y) ; Shadows top-level x.
;     (λ (x) ; Shadows that x.
;       (+ x y)))
;   11 7)
;  13)

(define change 0)
(define make-constant
  (λ (y)
    (set! change (λ () (set! y 10)))
    (λ () y)))
(define c8 (make-constant 8))
; Seems equivalent to returning (λ () 8).
(c8)
(change) ; changes c9's closure's environment
(c8) ; But it isn't.
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
