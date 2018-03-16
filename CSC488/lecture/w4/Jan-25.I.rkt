#lang racket #| Review of Curried Definitions, Racket Macros, and Church Encoding of Booleans |#

#;(define (f x) (+ x 1))
; means
#;(define f (λ (x) (+ x 1)))

; That generalizes:
#;(define ((g x) y) (+ (* 2 x) (* 3 y)))
; means
#;(define (g x) (λ (y) (+ (* 2 x) (* 3 y))))
; means
#;(define g (λ (x) (λ (y) (+ (* 2 x) (* 3 y)))))
; Notice that ((g x) y) is also a form in which we can use g:
#;((g 40) 500)
; But we don't have to call the result of g immediately:
#;(map (g 40) (list 500 600 700))

; Let's redefine ‘+’ to refer to a curried binary addition, and make function definitions produce
;  curried functions [which is how function definition behaves in, e.g., Haskell, OCaml, and F#].

; First, re-import addition and definition with the names ‘racket:+’ and ‘racket:define’.
(require (only-in racket [+ racket:+] [define racket:define]))

; A racket macro: *re-arranges* code in the rest of the file *at compile time*.
(define-syntax define ; Specify how to rearrange code of the form: (define <stuff> ...)
  (syntax-rules () ; One of the code re-arrangement sub-languages.

    ; Clauses: pairs of code pattern and rewrite template.
    
    ; Everything except the parenthesized structure, and the ‘define’ identifier at the front,
    ;  is taken non-literally.
    [(define (f-id id) body)
     ; Every non-literal identifier from the pattern is substituted where it appears here,
     ;  and all other identifiers [‘define’ and ‘λ’ in this case] are treated literally.
     (racket:define f-id (λ (id) body))]

    [(define (f-id id ... id-last) body)
     (define (f-id id ...) (λ (id-last) body))]))

(define (+ x y) (racket:+ x y))
; That turns into, at compile time
#;(define (+ x) (λ (racket:+ x y)))
; which turns into
#;(define + (λ (x) (λ (y) (racket:+ x y))))

((+ 40) 500)
(map (+ 40) (list 500 600 700))

; Church Booleans
; ===============
; What matters about booleans is making choices based on them.
; Can we make the following logic work, using only the LC?
#;(if (zero? 0) 123 (/ 456 0))

; That's doomed if ‘if’ is an eager function.
(require (only-in racket [if racket:if]))
#;(define (if c t e) (racket:if c t e)) ; By the time it executes the body, it's too late.

; Most important concept: to delay/select/control evaluation, create a function out of the computation
;  you want to control, and selectively call it.
#;(racket:if (zero? 0) ((λ () 123)) ((λ (/ 456 0))))

; This works [except the call to ‘if’ would have to adjust to the currying of the current ‘define’].
#;(define (if c t e) (racket:if c (t) (e)))
#;(if (zero? 0) (λ () 123) (λ () (/ 456 0)))

; Church boolean: call one of two functions.
#;(define (true t e) (t))
#;(define (false t e) (e))
#;(define (if c t e) ((c t) e))
#;(if (zero? 0) (λ () 123) (λ () (/ 456 0)))

; Let's put that all together, with only unary functions.
; Conditionals need to be rewritten at compile time to wrap their branches in λs.
(define-syntax-rule (if c-expr t-expr e-expr)
  ((c-expr (λ (_) t-expr)) (λ (_) e-expr)))
(define (true t e) (t 0)) ; The ‘0’ is arbitrary, t and e won't use their argument.
(define (false t e) (e 0))

((true (λ (_) 123)) (λ (_) (/ 456 0)))
((false (λ (_) (/ 456 0))) (λ (_) 123))

(if true 123 (/ 456 0))
(if false (/ 456 0) 123)

; Let's bring numeric comparison into our world.
(require (only-in racket [< racket:<]))
(define (< x y) (racket:if (racket:< x y) true false))

(if ((< 1) 2) 3 4)
(if ((< 2) 1) 3 4)
