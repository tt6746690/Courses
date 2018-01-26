#lang racket #| Compile Language L0 to Language L1 |#

(provide debruijn index-λs L0→L1)
(module+ test (require rackunit))

#| Language L0 is a slightly enriched version of the LC.

 An L0 expression is defined structurally [recall CSC236 which covered reading such definitions].
 It's one of: |#
#;(L0: λ (<id>) <e>)
#;(L0: app <e1> <e2>)
#;(L0: var <id>)
#;(L0: datum <i>)
#;(L0: set! <id> <e>)
;  where <e>, <e1>, <e2> are L0 expressions, <id> is an identifier, and <i> is an integer.

; As shown, in this assignment we'll tag language terms with an initial symbol indicating which
;  language they're from.

; As usual, our runtime representation of such expressions is racket lists for parenthesized terms,
;  and racket symbols for names.

; This isn't the first few weeks of CSC108. You're months away from industry or graduate school,
;  so you automatically and systematically generate examples of such expressions, to make the rest of
;  this description concrete and meaningful. You also do this because you follow industry standard
;  Test Driven Development.
; We're still happy to help you develop these skills if you've managed to avoid that until now, but
;  can't indulge requests that actively avoid learning those skills where they are required ---
;  that's a waste of everyone's time in a course that is creating a modern compiler for a modern
;  language down to a real architecture.

#| Semantics of language L0 |#
#;(L0: λ (<id>) <e>)  ; Unary closure creation.
#;(L0: app <e1> <e2>) ; Unary function call.
#;(L0: var <id>)      ; Variable access.
#;(L0: datum <i>)     ; Integer constant.
#;(L0: set! <id> <e>) ; Variable mutation: set <id> to the result of <e>.


#| Language L1 |#

; In the following, <n> is a natural number.
#;(L1: λ <n> <e>)     ; The nth lambda in an L1 expression.
#;(L1: app <e1> <e2>) ; Same meaning as in L0.
#;(L1: var <n>)       ; Reference to a variable <n> scopes up.
#;(L1: var <id>)      ; Free/unbound/open variable reference to <id>.
#;(L1: set! <n> <e>)  ; Set the variable <n> scopes up to the value of <e>.
#;(L1: datum <i>)     ; Same meaning as in L0.


#| DeBruijn Indexing of an L0 Expression

 Replaces each variable referencing a parameter, with a natural number indicating how many scopes up
  the variable is. Free variables are left alone: in later passes they will turn into references to
  pre-defined functions.

 Leaves the form of other L0 expressions alone, but does rewrite their sub-expressions.

 The optional argument env is a simple symbol table: it is the list of all ancestor parameters,
  with their positions being an implicit mapping to how far up in scope they are. |#

(define (debruijn e [env '()]) ; Takes an optional second argument, which defaults to the empty list.
  (index-of env e))


#| Indexing λs of a Debruijnized L0 Expression

 Replaces each L0 λ with an L1 λ, replacing the parameter list with a numeric index.
 Indexing starts at the value produced by the optional counter argument count, and is applied
  post-order when considering the expression as a tree. |#

; A class to make counting objects.
(define ((counter [c 0]))
  (set! c (add1 c))
  (sub1 c))

(module+ test
  (define c (counter))
  (check-equal? (c) 0)
  (check-equal? (c) 1)
  (define c′ (counter))
  (check-equal? (c′) 0)
  (check-equal? (c) 2))

; For a debruijned L0 expression, give each λ expression in e a unique index.
(define (index-λs e [count (counter)])
  (count))


#| L0→L1

 For an L0 expression: debruijnizes, indexes λs, and replaces remaining ‘L0:’ tags with ‘L1:’. |#

(define (L0→L1 e)
  (define (L0→L1′ e) e)
  (L0→L1′ (index-λs (debruijn e))))
