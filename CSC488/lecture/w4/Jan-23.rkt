#lang racket

(require "tree.rkt")

; Let's parse nested function calls and variable access.

#;"a"
#;"abc"
#;"a()"
#;"a(b)"
#;"a(b,c)"
#;"a(b(c),d)"

#| Let's make a “Context Free Grammar” for these.

 expr =   identifier
        | identifier "(" ")"
        | identifier "(" arguments ")"

 identifier =   alphabetic
              | alphabetic identifier

  arguments =   expr
              | expr "," arguments

  alphabetic we leave implicit: an alphabetic character

  expr, identifier, arguments, alphabetic are “non-terminals”
  "(" ")" "," "a", "b", "A", ... are “terminals” |#

(current-font-size 12)
#;"abc"
(tree '(expr (identifier (alphabetic "a")
                         (identifier (alphabetic "b")
                                     (identifier (alphabetic "c"))))))
#;"a(b,c)"
(tree '(expr (identifier (alphabetic "a"))
             "("
             (arguments (expr (identifier (alphabetic "b"))) ","
                        (arguments (expr (identifier (alphabetic "c")))))
             ")"))

(struct parsed (result rest) #:transparent)

(module+ test
  (require rackunit)
  (check-equal? (alphabetic "") #false)
  (check-equal? (alphabetic "a") (parsed '(alphabetic "a") ""))
  (check-equal? (alphabetic "ab") (parsed '(alphabetic "a") "b"))
  (check-equal? (alphabetic "(") #false)
  (check-equal? (identifier "") #false)
  (check-equal? (identifier "a") (parsed '(identifier (alphabetic "a")) ""))
  (check-equal? (identifier "ab") (parsed '(identifier (alphabetic "a")
                                                       (identifier (alphabetic "b")))
                                          ""))
  (check-equal? (identifier "(") #false)
  (check-equal? (identifier "ab(") (parsed '(identifier (alphabetic "a")
                                                        (identifier (alphabetic "b")))
                                           "(")))

(define (expr s)
  (parsed #false ""))

;  identifier =   alphabetic
              ;   | alphabetic identifier
(define (identifier s)
  (match (alphabetic s)
    [(parsed the-alphabetic s)
     (match (identifier s)
       [(parsed the-identifier s)   ; recursive case: alphabetic identifier
        (parsed `(identifier ,the-alphabetic ,the-identifier) s)]
       [#false                      ; base case:      alphabetic
        (parsed `(identifier ,the-alphabetic) s)])]
    [#false #false]))

(define (string-first s) (substring s 0 1))
(define (string-rest s) (substring s 1))

; s is alphabetic if
;    1. s not empty string
;    2. s is alphabetic
(define (alphabetic s)
  (and (not (equal? s ""))
       (char-alphabetic? (string-ref s 0))
       (parsed `(alphabetic ,(string-first s)) (string-rest s))))
