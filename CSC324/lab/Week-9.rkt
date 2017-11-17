#lang racket #| Programming with Generators |#

#| See the lectures under Oct 25 and Nov 1. |#

(require (only-in racket/control call/comp prompt abort)
         rackunit)

; Name the continuation of the whole let-continuation expression,
;  up to the most recently entered prompt,
;  and evaluate the body expressions in order,
;  using the last one for the result value.
(define-syntax-rule (let-continuation id body ...) (call/comp (λ (id) body ...)))


; ‘yield’ suspends a computation up to the most recently entered prompt, returning a list with:
;   1. The argument to ‘yield’.
;   2. A thunk [no-argument function] for resuming the computation, that also protects the caller
;       from future yield aborts by wrapping the continuation in a prompt.
(define (yield a-result)
  (let-continuation resume
                    (abort (list a-result
                                 (λ () (prompt (resume)))))))

(define value first) ; Yielded value.

; ★ Fix the implementation of resume to call the part of ‘yielded’ that resumes computation.
;
; See the example usage that follows it, including the printed result of calling ‘f’.
;
(define (resume yielded)
  ((second yielded))) ; Resume a yielded generator.
;
(define (f)
  (prompt (yield 1)
          (yield 2)))
;
(define y1 (f))  
; Contains the yielded value and a function to resume computation:
(println y1)    ; (1, resume)
; The yielded value is 1:
(check-equal? (value y1) 1)
; Resume to after the first yield:
(define y2 (resume y1))
; Should contain the yielded value 2 and a new function to resume computation.
(println y2)
(check-equal? (value y2) 2)


; ★ Fix the implemetation of ‘define*’, a simple version of Javascript's ‘function*’
;    for defining a generator function: a function that can yield to act as an iterator.
;
; See the example usage for ‘atoms’ that follows it.
;
(define-syntax-rule (define* (f-id p-id ...)
                      body
                      ...)
  (define (f-id p-id ...)
    (define (f-id p-id ...) body...))
  (prompt (f-id p-id ...)
          #false))
;
; ‘flatten’ as a visitor, without the gathering
; ‘for-each’ is like ‘map’, but doesn't accumulate the results: it's used only for the side-effects.
;     i.e. for-each returns void
(define* (atoms v)
  (cond [(list? v) (for-each atoms v)]
        [else (yield v)]))
;
; Intended expansion:
;
#;(define (atoms v)
    ; Locally define the function so we can do something before it starts,
    ;  without affecting any recursive calls.
    (define (atoms v)
      (cond [(list? v) (for-each atoms v)]   
            [else (yield v)]))
    ; Wrap the call to the function in a prompt so that control flow manipulation is explicitly about
    ;  the continuation up to just before returning to the client of ‘atoms’, and not about anything
    ;  involving the client's code.
    ; Return #false after all yields, to distinguish normal return from resumable yielded returns.
    (prompt (atoms v)
            #false))
;
(define ya (atoms '(a ((b) c))))
(check-equal? (value ya) 'a)
(define yb (resume ya))
(check-equal? (value yb) 'b)
(define yc (resume yb))
(check-equal? (value yc) 'c)
(define yend (resume yc))
(check-equal? yend #false)


; ★ Fix the implemetation of ‘to-list’, which gathers all yielded values of a computation as a list.
;
; See the example usage for ‘to-list’ that follows it.
;
(define (to-list yielded)
  '())
;
(check-equal? (to-list (atoms '(a ((b) c) (d (e)))))
              '(a b c d e))


; Recall ‘adjacent’ and ‘paths’ from earlier in the course.
; ★ Fix ‘path’, to yield the path when a path is found.
; The meaning of the ‘froms’ parameter is now the accumulated “path so far”, with “from” at the front.
;
(define (adjacent spot)
  (map (λ (Δ) (map + spot Δ)) '((0 -1) (0 1) (-1 0) (1 0))))
;
(define* (path to through froms)
  (define from (first froms))
  (yield froms))
;
(define figure-eight (list (list 0 0) (list 1 0) (list 2 0)
                           (list 0 1)            (list 2 1)
                           (list 0 2) (list 1 2) (list 2 2)
                           (list 0 3)            (list 2 3)
                           (list 0 4) (list 1 4) (list 2 4)))
;
(check-equal? (to-list (path (list 2 4) figure-eight (list (list 0 0))))
              '(((2 4) (1 4) (0 4) (0 3) (0 2) (0 1) (0 0))
                ((2 4) (2 3) (2 2) (1 2) (0 2) (0 1) (0 0))
                ((2 4) (2 3) (2 2) (2 1) (2 0) (1 0) (0 0))
                ((2 4) (1 4) (0 4) (0 3) (0 2) (1 2) (2 2) (2 1) (2 0) (1 0) (0 0))))
