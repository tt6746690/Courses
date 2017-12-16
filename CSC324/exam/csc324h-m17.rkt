#lang racket 


; Q1.A
(apply append (list (list (list 1 2) (list 3 4))
                    (list (list 5 6) (list 7 8))))

(map append '((1 2) (3 4)) '((5 6) (7 8)))


; Q1.B
(define (R LoL)
  (cond [(list? LoL) (reverse (map R LoL))]
        [else LoL]))
(define (RA LoL)
  (cond [(list? LoL) (reverse (apply append (map RA LoL)))]
        [else (list LoL)]))


(R '((1 (2 3))))
(RA '((1 2) 3 4))



; Q2

; (require "mm.rkt")
; (wait! #false)
; 
; (define (Counter i)
;   (λ (d)
;     (set! i (+ d i))
;     i))
; 
; (define c (Counter 0))
; (map c '(1 20))
; ; (1 21)
; 
; (map (λ (i) ((Counter i) i)) '(300 4000))
; ; '(600 8000)

; Q5
(define (implies b0 b1) (or (not b0) b1))
(define (sorted? L)
  (implies (>= (length L) 2)
           (and (< (first L) (second L))
                (sorted? (rest L)))))

; (sorted? '())

; Q7
(require "amb.rkt") ; Backtracking library.w9

(define-syntax when-assert
  (syntax-rules ()
    [(when-assert <condition> <body> ...)
     (cond [(equal? #t <condition>) <body> ...]
           [else (fail)])]))

(define-syntax cond-<
  (syntax-rules ()
    [(cond-< [<condition> <body> ...]
             ...)
     (-< (when-assert <condition> <body> ...) ...)]))


(define (generate-term recursions)
  (local [(define n (- recursions 1))]
    (cond-< [(zero? recursions) 'sin]
            [(zero? recursions) 'cos]
            [(zero? recursions) 'exp]
            [(> recursions 0) `(,(generate-term n) + ,(generate-term n))]
            [(> recursions 0) `(,(generate-term n) x ,(generate-term n))]
            [(> recursions 0) `(,(generate-term n) ∘ ,(generate-term n))])))


(module+ test 
  (require rackunit)
  (check-equal? (list-results (cond-< [(< 1 2) 3]
                                      [(> 1 2) (println 'hi) 4]
                                      [(zero? 0) (println 'totally) 5]))
                '(3 5))
  (check-equal? (list-results (generate-term 0))
                '(sin cos exp))
  (check-equal? (list-results (generate-term 1))
                '((sin + sin)
                  (sin + cos)
                  (sin + exp)
                  (cos + sin)
                  (cos + cos)
                  (cos + exp)
                  (exp + sin)
                  (exp + cos)
                  (exp + exp)
                  (sin x sin)
                  (sin x cos)
                  (sin x exp)
                  (cos x sin)
                  (cos x cos)
                  (cos x exp)
                  (exp x sin)
                  (exp x cos)
                  (exp x exp)
                  (sin ∘ sin)
                  (sin ∘ cos)
                  (sin ∘ exp)
                  (cos ∘ sin)
                  (cos ∘ cos)
                  (cos ∘ exp)
                  (exp ∘ sin)
                  (exp ∘ cos)
                  (exp ∘ exp)))
  )


