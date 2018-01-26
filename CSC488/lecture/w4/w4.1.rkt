#lang racket

(require "tree.rkt")
;(current-font-size 20)

; lets parse nested function calls and variable access
"a"
"abc"
"a()"
"a(b)"
"a(b,c)"
"a(b(c),d)"


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

; parse tree for "abc"
(tree '(expr (identifier (alphabetic "a")
                         (identifier (alphabetic "b")
                                     (identifier (alphabetic "c"))))))
; parse tree for "a(b,c)"
(tree '(expr (identifier (alphabetic "a"))
             "("
             (arguments (expr (identifier (alphabetic "b"))) ","
                        (argument (expr (identifier (alphabetic "c")))))
             ")"))
; note parenthesis does not matter, since if
;   expr has 1 child implies variable access
;   expr has 2 child implies function call


(require rackunit)
(struct parsed (result rest) #:transparent)
; constructor: (parsed fields ...)
; result is either the parsed result or #false, if cannot be parsed


; recursive descend
; make a set of mutually recursive functions that recognize
; non-terminals in the expression
(module+ test
  (check-equal? (expr "a")
                (parsed '(expr (identifier (alphabetic "a"))) ""))
  ; alphabetic
  (check-equal? (alphabetic "")
                #f)
  (check-equal? (alphabetic "a")
                (parsed '(alphabetic "a") ""))
  (check-equal? (alphabetic "ab")
                (parsed '(alphabetic "a") "b"))
  (check-equal? (alphabetic "(")
                #f)
  ; propagate #f
  (check-equal? (identifier "(")
                #f)

  ; identifier
  (check-equal? (identifier "ab")
                (parsed
                 '(identifier (alphabetic "a") (identifier (alphabetic "b")))
                 "") )
  )

(define (expr s)
  (parsed #false ""))


; identifier =   alphabetic
;              | alphabetic identifier
(define (identifier s)
  (match (alphabetic s)  
    [(parsed the-alphabetic s)  ; pattern matching against struct 
     (match (identifier s)
       [(parsed the-identifier s)
        ; alphabetic identifier
        (parsed `(identifier ,the-alphabetic ,the-identifier) s)]
       [#false
        ; alphabetic
        (parsed `(identifier ,the-alphabetic) s)])
     ]
    [#false #false]))

(define (string-first s) (substring s 0 1))
(define (string-rest s) (substring s 1))

; return #false if s cannot be parsed
(define (alphabetic s)
  (and (not (equal? s ""))
       (char-alphabetic? (string-ref s 0))
       (parsed `(alphabetic ,(string-first s)) (string-rest s))))


; Lets work on literals and sequencing
; Temporarily: argument-group = "(" alphabetic ")"


#; (define (left-parenthesis s)
     (and (not (equal? s ""))
          (equal? (string-first s) "(")
          (parsed (string-first s) (string-rest s))))


#; (define (left-parenthesis s)
     (and (not (equal? s ""))
          (equal? (string-first s) ")")
          (parsed (string-first s) (string-rest s))))

(define ((literal s0) s)
  (and (not (equal? s ""))
       (equal? (string-first s) s0)
       (parsed (string-first s) (string-rest s))))

; parser for ( and )
(define left-parenthesis (literal "("))
(define right-parenthesis (literal ")"))

; take 2 parser
(define ((sequence p1 p2) s)
  (match (p1 s)
    [(parsed p1-result s)
     (match (p2 s)
       [(parsed p2-result s)
        (parsed `(sequence ,p1-result ,p2-result) s)]
       [#false #false])]
    [#false #false]))

(define bracket (sequence left-parenthesis right-parenthesis))



