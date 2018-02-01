#lang racket

(require "tree.rkt")

#| Lets make a "Context Free Grammar" for this

 expr =   identifier                    ; variable access
        | identifier "(" ")"            ; function call
        | identifier "(" arguments ")"
 · order of clauses is a problem, really should be the reverse order

 identifier =   alphabetic
              | alphabetic identifier

 ; or
 identifier = alphabetic [identifier]

 arguments =   expr
             | expr "," arguments
 · arguments = expr | arguments "," expressions, will recurse indefinitely

 ; or
 arguments = expr ["," expr]*

 expr, identifier, arguments, alphabetic are "non-terminals"
 "(", ")", :,", "a", "b", "A", ⋯ are "terminals" (base case)
|#



(require rackunit)
(struct parsed (result rest) #:transparent)

(define (string-first s) (substring s 0 1))
(define (string-rest s) (substring s 1))

; return #false if s cannot be parsed
(define (alphabetic s)
  (and (not (equal? s ""))
       (char-alphabetic? (string-ref s 0))
       (parsed `(alphabetic ,(string-first s)) (string-rest s))))


(define ((literal s0) s)
  (and (not (equal? s ""))
       (equal? (string-first s) s0)
       (parsed (string-first s) (string-rest s))))

; parser for ( and )
(define left-parenthesis (literal "("))
(define right-parenthesis (literal ")"))

; take 2 parser
#;(define ((sequence p1 p2) s)
    (match (p1 s)
      [(parsed p1-result s)
       (match (p2 s)
         [(parsed p2-result s)
          (parsed `(sequence ,p1-result ,p2-result) s)]
         [#false #false])]
      [#false #false]))


#; (parse-when (p2 s) (p2 result s)
               (parsed `(sequence ,p1-result ,p2-result) s))
#; (match (p2 s)
     [(p2 s) (p2 result s)
             (parsed `(sequence ,p1-result ,p2-result) s)])

(define-syntax-rule
  (parse-when expr (result-id rest-id) ; name variable output of (p1 s), which is a parsed
              result-expr)             ; expr that run if successfully parsed
  (match expr
    [(parsed result-id rest-id)
     result-expr]
    [#false #false]))

#;(define ((sequence p1 p2) s)
    (parse-when (p1 s) (p1-result s)  
                (match (p2 s)         
                  [(parsed p2-result s)
                   (parsed `(sequence ,p1-result ,p2-result) s)]
                  [#false #false])))


(define ((sequence p1 p2) s)
  (parse-when (p1 s) (p1-result s)
              (parse-when (p2 s) (p2-result s)
                          (parsed `(sequence ,p1-result ,p2-result) s ))))


(define bracket (sequence left-parenthesis right-parenthesis))
#|
  expression = identifier [argument-group]
  argument-group = "(" [arguments] ")"
  arguments = expression ["," arguments]*
  identifier = alphabetic [identifier]
|#

; ((maybe (literal "a")) "abc")  ;  (parsed "a" "bc")
; ((maybe (literal "a")) "bc")   ;  (parsed 'ε "bc")
(define ((maybe p) s)
  (or (p s)
      (parsed 'ϵ s)))

#; (define identifier
     (sequence alphabetic (maybe identifier)))
; wrong since eager evaluation, (maybe identifer) evaluated,
; identifier evaluated recursively

#;(define (identifier s)
    ((sequence alphabetic (maybe identifier)) s))

(define-syntax-rule (define-recursive f-id body-expr)
  (define (f-id v) (body-expr v)))

(define-recursive identifier
  (sequence alphabetic (maybe identifier)))
; (tree (parsed-result (identifier "abc")))


; (flatten-sequence (parsed-result (identifier "abc")))
(define (flatten-sequence a-sequence)
  (match a-sequence
    ['ϵ '()]
    [`(sequence ,left ,right)
     `(,left . ,(flatten-sequence right))]))


