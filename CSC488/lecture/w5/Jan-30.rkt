#lang racket

#| expression = identifier [argument-group]
   argument-group = "(" [arguments] ")"
   arguments = expression ["," arguments]*
   identifier = [alphabetic]+ |#

(define (alphabetic? s) (char-alphabetic? (string-ref s 0)))
(define (string-first s) (substring s 0 1))
(define (string-rest s) (substring s 1))
(define (string-empty? s) (equal? s ""))

(struct parsed (result rest) #:transparent)

(define (alphabetic s)
  (and (not (equal? s ""))
       (char-alphabetic? (string-ref s 0))
       (parsed `(alphabetic ,(string-first s)) (string-rest s))))

(define ((literal s0) s)
  (and (not (equal? s ""))
       (equal? (string-first s) s0)
       (parsed (string-first s) (string-rest s))))

(define left-parenthesis (literal "("))
(define right-parenthesis (literal ")"))

#;(parse-when (p2 s) (p2-result s)
              (parsed `(sequence ,p1-result ,p2-result) s))
#;(match (p2 s)
    [(parsed p2-result s)
     (parsed `(sequence ,p1-result ,p2-result) s)]
    [#false #false])

(define-syntax-rule
  (parse-when expr (result-id rest-id)
              result-expr)
  (match expr
    [(parsed result-id rest-id)
     result-expr]
    [#false #false]))

#;(define (sequence p1 p2)
    (λ (s)
      (match (p1 s)
        [(parsed p1-result s)
         (match (p2 s)
           [(parsed p2-result s)
            (parsed `(sequence ,p1-result ,p2-result) s)]
           [#false #false])]
        [#false #false])))

(define ((sequence p1 p2) s)
  (parse-when (p1 s) (p1-result s)
     (parse-when (p2 s) (p2-result s)
        (parsed `(sequence ,p1-result ,p2-result) s))))

#| expression = identifier [argument-group]
   argument-group = "(" [arguments] ")"
   arguments = expression ["," arguments]*
   identifier = alphabetic [identifier] |#

(define ((maybe p) s)
  (or (p s)
      (parsed 'ϵ s)))

#;(define identifier
    (λ (s) ((sequence alphabetic (maybe identifier)) s)))
; need to create a thunk since (maybe identifier) evaluated first 

(define-syntax-rule (define-recursive f-id body-expr)
  (define (f-id v) (body-expr v)))

#; (define (identifier v)
     ((sequence alphabetic (maybe identifier)) v))

(define-recursive identifier
  (sequence alphabetic (maybe identifier)))
(require "tree.rkt")
(current-font-size 24)

(define (flatten-sequence a-sequence)
  (match a-sequence
    ['ϵ '()]
    [`(sequence ,left ,right)
     (list* left (flatten-sequence right))
     #;(list left (flatten-sequence right))
     #;`(,left . ,(flatten-sequence right))]))
