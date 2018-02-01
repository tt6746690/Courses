#lang racket #| Compile Language L0 to Language L1 |#

(provide debruijn index-λs L0→L1)
(module+ test (require rackunit))
(require "tree.rkt")

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

(define (push stack x)
  (append `(,x) stack))

(define (pop stack)
  (match stack
    [(list* top rest) rest]
    [_ stack]))

(module+ test
  ; test push/pop
  (check-equal? (local [(define l '(1 2 3))] (push l 0))
                '(0 1 2 3))
  (check-equal? (local [(define l '(1 2 3))] (pop l))
                '(2 3))
  ; empty case
  (check-equal? (debruijn '())
                '())
  ; test var
  (check-equal? (debruijn '(L0: λ (x) (L0: var x)))    ; 0 scope up
                '(L0: λ (x) (L1: var 0)))
  (check-equal? (debruijn '(L0: λ (x) (L0: var y)))    ; free variable unmodified
                '(L0: λ (x) (L1: var y)))
  (check-equal? (debruijn '(L0: λ (x) (L0: app (L0: var x)
                                           (L0: λ (y) (L0: app (L0: var y) (L0: var x))))))  ; 2 levels
                '(L0: λ (x) (L0: app (L1: var 0)
                                 (L0: λ (y) (L0: app (L1: var 0) (L1: var 1))))))
  ; test set!
  (check-equal? (debruijn '(L0: λ (x) (L0: set! x (L0: datum 12))))
                '(L0: λ (x) (L1: set! 0 (L0: datum 12))))
  (check-equal? (debruijn '(L0: λ (x) (L0: set! x (L0: λ (y) (L0: set! x (L0: var y))))))
                '(L0: λ (x) (L1: set! 0 (L0: λ (y) (L1: set! 1 (L1: var 0))))))
  (check-equal? (debruijn '(L0: λ (x) (L0: set! x (L0: λ (y) (L0: set! p (L0: var z)))))) ; free variable unmodifed
                '(L0: λ (x) (L1: set! 0 (L0: λ (y) (L1: set! p (L1: var z))))))

  ; test env are separate for different recursive cases
  (check-equal? (debruijn '(L0: app (L0: λ (x) (L0: var x)) (L0: λ (y) (L0: var x))))
                '(L0: app (L0: λ (x) (L1: var 0)) (L0: λ (y) (L1: var x))))
  )


; add current arg when creating a closure
; update (L0: var <id>) → (L1: var <n>)
;        (L0: set! <id> <e>) → (L1: set! <n> <e>)
(define (debruijn e [env '()]) ; Takes an optional second argument, which defaults to the empty list.
  (define (debruijn-index id)
    (or (index-of env id) id))
  (match e
    [`(L0: λ (,<id>) ,<e>) `(L0: λ (,<id>) ,(debruijn <e> (push env <id>)))]  ; only case to push to env
    [`(L0: app ,<e1> ,<e2>) `(L0: app ,(debruijn <e1> env) ,(debruijn <e2> env))]
    [`(L0: var ,<id>) `(L1: var ,(debruijn-index <id>))]
    [`(L0: set! ,<id> ,<e>) `(L1: set! ,(debruijn-index <id>) ,(debruijn <e> env))]
    [_ e]))


#| Indexing λs of a Debruijnized L0 Expression

 Replaces each L0 λ with an L1 λ, replacing the parameter list with a numeric index.
 Indexing starts at the value produced by the optional counter argument count, and is applied
  post-order when considering the expression as a tree. |#

; A class to make counting objects.
(define ((counter [c 0]))
  (set! c (add1 c))
  (sub1 c))

(module+ test
  ; counter
  (define c (counter))
  (check-equal? (c) 0)
  (check-equal? (c) 1)
  (define c′ (counter))
  (check-equal? (c′) 0)
  (check-equal? (c) 2)

  ; test λ
  (check-equal? (index-λs '(L0: λ (x) (L1: var 0)))
                '(L1: λ 0 (L1: var 0)))
  (check-equal? (index-λs '(L0: λ (x) (L0: λ (y) (L1: var 1))))
                '(L1: λ 1 (L1: λ 0 (L1: var 1))))
  ; test app
  (check-equal? (index-λs '(L0: app (L0: λ (x) (L1: var 0)) (L0: datum 0)))
                '(L0: app (L1: λ 0 (L1: var 0)) (L0: datum 0)))
  (check-equal? (index-λs '(L0: app (L0: λ (x) (L1: var 0)) (L0: λ (y) (L1: var 0))))
                '(L0: app (L1: λ 0 (L1: var 0)) (L1: λ 1 (L1: var 0))))
  (check-equal? (index-λs '(L0: λ (x) (L0: app (L0: λ (y) (L1: var 1)) (L0: λ (z) (L1: var 1)))))
                '(L1: λ 2 (L0: app (L1: λ 0 (L1: var 1)) (L1: λ 1 (L1: var 1)))))  ; check post order

  ; test set!
  (check-equal? (index-λs '(L1: set! 0 (L0: λ (x) (L1: var 0))))
                '(L1: set! 0 (L1: λ 0 (L1: var 0))))
  (check-equal? (index-λs '(L0: λ (x) (L1: set! 0 (L0: λ (y) (L1: var 1)))))
                '(L1: λ 1 (L1: set! 0 (L1: λ 0 (L1: var 1)))))
  ; test post order complex case
  (check-equal? (index-λs '(L0: λ (x) (L1: set! 0 (L0: app (L0: λ (y) (L1: var 1)) (L0: λ (z) (L1: var 1))))))
                '(L1: λ 2 (L1: set! 0 (L0: app (L1: λ 0 (L1: var 1)) (L1: λ 1 (L1: var 1))))))
  (check-equal? (index-λs '(L0: λ (x) (L1: set! 0 (L0: app (L0: λ (y) (L1: var 1)) (L0: λ (z) (L0: λ (m) (L1: var 1)))))))
                '(L1: λ 3 (L1: set! 0 (L0: app (L1: λ 0 (L1: var 1)) (L1: λ 2 (L1: λ 1 (L1: var 1)))))))
  
  )

;(tree '(λ (x) (set! 0 (app (λ (y) (var 1)) (λ (z) (λ (m) (var 1)))))))
;(tree '(λ 3 (set! 0 (app (λ 0 (var 1)) (λ 2 (λ 1 (var 1)))))))

; Debruijnized L0 expression, 
#;(L0: λ (<id>) <e>)     ; The nth lambda in an L1 expression.
#;(L0: app <e1> <e2>) ; Same meaning as in L0.
#;(L1: var <n>)       ; Reference to a variable <n> scopes up.
#;(L1: var <id>)      ; Free/unbound/open variable reference to <id>.
#;(L1: set! <n> <e>)  ; Set the variable <n> scopes up to the value of <e>.
#;(L0: datum <i>)     ; Same meaning as in L0.

; For a debruijned L0 expression, give each λ expression in e a unique index.
; update: (L0: λ (<id>) <e>) → (L1: λ <n> <e>)
; recursive call index-λs for app, set!
(define (index-λs e [count (counter)])
  (match e
    [`(L0: λ (,<id>) ,<e>) (define post-order-e (index-λs <e> count))
                           `(L1: λ ,(count) ,post-order-e)]
    [`(L0: app ,<e1> ,<e2>) `(L0: app ,(index-λs <e1> count) ,(index-λs <e2> count))]
    [`(L1: set! ,<n> ,<e>) `(L1: set! ,<n> ,(index-λs <e> count))]
    [_ e]))


#| L0→L1

 For an L0 expression: debruijnizes, indexes λs, and replaces remaining ‘L0:’ tags with ‘L1:’. |#


(module+ test
  ; test conversion from L0 → L1 for app and datum
  (check-equal? (L0→L1 '(L0: datum 10))
                '(L1: datum 10))
  (check-equal? (L0→L1 '(L0: app (L0: λ (x) (L0: var x)) (L0: λ (y) (L0: var x))))
                '(L1: app (L1: λ 0 (L1: var 0)) (L1: λ 1 (L1: var x))))
  )
(define (L0→L1 e)
  (define (L0→L1′ e)
    (match e
      [`(L0: ,<syntax> ...) `(L1: . ,<syntax>)]
      [_ e]))
  (L0→L1′ (index-λs (debruijn e))))
