#lang racket #| Compile A2's Language L0 to A2's Language L1 |#

(provide debruijn index L0→L1)
(module+ test (require rackunit))

#| A2's language L0 is A1's language L0 with one additional conditional expression. |#

; If <e1> is true then evaluate <e2>, else evaluate <e3>.
#;(L0: if <e1> <e2> <e3>)

#| A2's language L1 is A1's language L1 with one additional conditional expression. |#

; The nth if expression.
#;(L1: if <n> <e1> <e2> <e3>)

#| DeBruijn Indexing of an L0 Expression

 This is the same as in A1. |#

#;(define (debruijn e [env '()]) ; Takes an optional second argument, which defaults to the empty list.
    (define (debruijn′ e) (debruijn e env))
    e)


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
  (define (debruijn′ e) (debruijn e env))
  (define (debruijn-index id)
    (or (index-of env id) id))
  (match e
    [`(L0: λ (,<id>) ,<e>) `(L0: λ (,<id>) ,(debruijn <e> (push env <id>)))]  ; only case to push to env
    [`(L0: app ,<e1> ,<e2>) `(L0: app ,(debruijn′ <e1>) ,(debruijn′ <e2>))]
    [`(L0: var ,<id>) `(L1: var ,(debruijn-index <id>))]
    [`(L0: set! ,<id> ,<e>) `(L1: set! ,(debruijn-index <id>) ,(debruijn′ <e>))]
    [_ e]))




#| Indexing of a Debruijnized L0 Expression

 For the A1 subset of L0 this is the same.
 The new conditional expressions are also given unique indices. |#


(module+ test
  ; counter
  (define c (counter))
  (check-equal? (c) 0)
  (check-equal? (c) 1)
  (define c′ (counter))
  (check-equal? (c′) 0)
  (check-equal? (c) 2)

  ; test λ
  (check-equal? (index '(L0: λ (x) (L1: var 0)))
                '(L1: λ 0 (L1: var 0)))
  (check-equal? (index '(L0: λ (x) (L0: λ (y) (L1: var 1))))
                '(L1: λ 1 (L1: λ 0 (L1: var 1))))
  ; test app
  (check-equal? (index '(L0: app (L0: λ (x) (L1: var 0)) (L0: datum 0)))
                '(L0: app (L1: λ 0 (L1: var 0)) (L0: datum 0)))
  (check-equal? (index '(L0: app (L0: λ (x) (L1: var 0)) (L0: λ (y) (L1: var 0))))
                '(L0: app (L1: λ 0 (L1: var 0)) (L1: λ 1 (L1: var 0))))
  (check-equal? (index '(L0: λ (x) (L0: app (L0: λ (y) (L1: var 1)) (L0: λ (z) (L1: var 1)))))
                '(L1: λ 2 (L0: app (L1: λ 0 (L1: var 1)) (L1: λ 1 (L1: var 1)))))  ; check post order

  ; test set!
  (check-equal? (index '(L1: set! 0 (L0: λ (x) (L1: var 0))))
                '(L1: set! 0 (L1: λ 0 (L1: var 0))))
  (check-equal? (index '(L0: λ (x) (L1: set! 0 (L0: λ (y) (L1: var 1)))))
                '(L1: λ 1 (L1: set! 0 (L1: λ 0 (L1: var 1)))))
  ; test post order complex case
  (check-equal? (index '(L0: λ (x) (L1: set! 0 (L0: app (L0: λ (y) (L1: var 1)) (L0: λ (z) (L1: var 1))))))
                '(L1: λ 2 (L1: set! 0 (L0: app (L1: λ 0 (L1: var 1)) (L1: λ 1 (L1: var 1))))))
  (check-equal? (index '(L0: λ (x) (L1: set! 0 (L0: app (L0: λ (y) (L1: var 1)) (L0: λ (z) (L0: λ (m) (L1: var 1)))))))
                '(L1: λ 3 (L1: set! 0 (L0: app (L1: λ 0 (L1: var 1)) (L1: λ 2 (L1: λ 1 (L1: var 1)))))))
  ; test if
  (check-equal? (index '(L0: if #t (L0: datum 1) (L0: datum 2)))
                '(L1: if 0 #t (L0: datum 1) (L0: datum 2)))
  (check-equal? (index '(L0: if
                             (L0: if #t (L0: datum 1) (L0: datum 2))
                             (L0: if #t (L0: datum 3) (L0: datum 4))
                             (L0: if #t (L0: datum 5) (L0: datum 6))))
                '(L1: if 0
                      (L1: if 1 #t (L0: datum 1) (L0: datum 2))
                      (L1: if 2 #t (L0: datum 3) (L0: datum 4))
                      (L1: if 3 #t (L0: datum 5) (L0: datum 6))))
  )

(define ((counter [c 0]))
  (set! c (add1 c))
  (sub1 c))

; For a debruijned L0 expression e, give each λ expression a unique index,
;  and each if expression a unique index.
(define (index e [λ-count (counter)] [if-count (counter)])
  (define (index′ e) (index e λ-count if-count))
  (match e
    [`(L0: λ (,<id>) ,<e>) (define post-order-e (index′ <e>))
                           `(L1: λ ,(λ-count) ,post-order-e)]
    [`(L0: app ,<e1> ,<e2>) `(L0: app ,(index′ <e1>) ,(index′ <e2>))]
    [`(L0: if ,<e1> ,<e2> ,<e3>) `(L1: if ,(if-count)
                                       ,(index′ <e1>)
                                       ,(index′ <e2>)
                                       ,(index′ <e3>))]
    [`(L1: set! ,<n> ,<e>) `(L1: set! ,<n> ,(index′ <e>))]
    [_ e]))


#;(L0: if <e1> <e2> <e3>)
#;(L1: if <n> <e1> <e2> <e3>)  ; The nth if expression.


#| L0→L1

 For an L0 expression: debruijnizes, indexes, and replaces remaining ‘L0:’ tags with ‘L1:’. |#

(module+ test
  ; test conversion from L0 → L1 for app and datum
  (check-equal? (L0→L1 '(L0: datum 10))
                '(L1: datum 10))
  (check-equal? (L0→L1 '(L0: app (L0: λ (x) (L0: var x))
                             (L0: λ (y) (L0: var x))))
                '(L1: app (L1: λ 0 (L1: var 0)) (L1: λ 1 (L1: var x))))
  )
(define (L0→L1 e)
  (define (L0→L1′ e)
    (match e
      [`(L0: ,<syntax> ...) `(L1: . ,<syntax>)]
      [_ e]))
  (L0→L1′ (index (debruijn e))))


