#lang racket #| Backtracking Library |#

(provide
 ; For users of backtracking code:
 stage trace next done? list-results
 ; For implementers of backtracking code:
 -< fail)

(define trace (make-parameter #true))

(require racket/control)

#| Our let-continuation. |#
(define-syntax-rule (let-continuation id body ...)
  (call/comp (λ (id) body ...)))

#| Stack of lists of delayed alternate expressions, to resume a point that control flowed through
    and use a different expression there, with the most recent delayed alternate first. |#
(define alternates (make-parameter '()))

#| Encapsualte a backtracking computation for the user.
   Clears pending alternates, and puts the initial expression delayed into the alternates. |#
(define-syntax-rule (stage e ...)
  (local []
    (alternates '())
    (announce 'stage '(e ...))
    (alternates (list (list (λ () e ...))))))

#| Unique value to return when there are no more alternates, with a predicate to detect it. |#
(define done (gensym 'done))
(define (done? v) (equal? v done))

#| Get the next value from an expression involving backtracking. |#
(define (next)
  (match (alternates)
    ['() (announce 'ended 'done) done]
    [`(() . ,alternates′) (alternates alternates′)
                          (next)]
    [`((,alternate . ,alternates′) . ,alternates′′) (alternates `(,alternates′ . ,alternates′′))
                                                    (prompt (alternate))]))

#| Abort part of a computation implemented with backtracking. |#
(define (fail)
  ; This calls next, then aborts with the result value.
  #;(abort (next))
  ; This version aborts and *then* calls next for the result value, which is more efficient
  ;  and easier to reason about.
  (announce 'fails 'fail)
  (abort-current-continuation
   (default-continuation-prompt-tag)
   next))

(define (announce action code)
  (when (trace)
    (printf "~a[~a]~a\n"
            (make-string (length (alternates)) #\space)
            action
            (if (or (symbol? code) (list? code))
                (substring (~v code) 1)
                (~v code)))))

#| Use one of the expressions e0 e ... for the value. |#
(define-syntax-rule (-< e0 e ...)
  (let-continuation retry
                    (alternates (list* (list (λ () (announce 'retry 'e) (retry e)) ...) (alternates)))
                    (announce 'using 'e0)
                    (announce 'aside 'e) ...
                    e0))

(define-syntax-rule (list-results e ...)
  (local [(define results (list))]
    (stage (set! results (list* (local [] e ...) results))
           (fail))
    (next)
    (reverse results)))
