#lang racket

(require "mm.rkt")
; Some parameters to change how the model is shown.
#;(scale! 20)
#;(body-width! 100)
#;(wait! #false)

; In the top-level environment.
(define count 0)

; Name in the top-level environment.
; Closure created under that environment.
(define (Counter)
  ; Each call to the function makes a new local scope.
  ; A variable is created each time, shadowing the top-level one.
  (define count 0)
  ; This function is made in that scope.
  (λ ()
    (set! count (add1 count))
    (sub1 count)))

; Calling Counter creates a local environment for its variables.
; The closure it creates and returns is made under that local environment.
(define c1 (Counter))
; Calling Counter again creates another local environment, and a closure under that
(define c2 (Counter))

; The two closures alter the count from the environment they were made in.
(c1)
(c1)
(c2)

; The local environments shadow the top-level one.
count
