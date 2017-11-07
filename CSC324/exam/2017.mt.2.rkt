#lang racket

;(require "mm.rkt")




; (define x 1)
; (define f (λ (x)
;             ; body 1
;             (λ ()
;               ; body 2
;               (set! x (+ x 20))
;               x)))
; 
; ((f 300))
; ((f 300))
; (define g (f 4000))
; 
; (g)
; (g)
; 
; 
; 
; (define (fix-1st f 1st) (λ (2nd) (f 1st 2nd)))
; 
; (define (inserter x) (λ (lst) (list
;                                (list* x lst)
;                                (unless (empty? lst) ; why unless
;                                  (map (fix-1st list* (first lst)) ((inserter x) (rest lst)))
;                                  ))))
; 
; 
; (check-equal? ((inserter 'x) '(a b c))
;               '((x a b c)
;                 (a x b c) (a b x c) (a b c x)))



; (define-syntax neither
;   (syntax-rules ()
;     [(neither <expr> ...) (and (not <expr>) ...)]))

(define (expand term)
  (match term
    [`(define (,f ,a) ,body)
     `(define ,f (λ (,a) ,body))]
    [_ term]))




