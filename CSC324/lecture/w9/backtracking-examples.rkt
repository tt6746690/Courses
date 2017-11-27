#lang racket #| Backtracking Examples. |#

(require "amb.rkt") ; Backtracking library.

; For sanity when using code that contains the backtracking operations.
#;(stage e
         ...)
; Inside there: use -< for choice, and fail to cut off a branch of computation.

; Tracing is on by default, you can turn it off with:
#;(trace #false)

(stage
 ; Display either "hi" or "bye".
 (displayln (-< "hi"
                ; "bye" isn't in the continuation of "hi"
                "bye")
            ; The display is in the continuation of (-< "hi" "bye"),
            ;  and so is in the continuation of "hi" when that's used,
            ;  and is in the continuation of "bye" when that's used.
            ; Each e in (-< e ...) is in tail position wrt to the expression.
            )
 ; That's more like an if, not like sequencing:
 #;(displayln (if (zero? (random 2))
                  "hi"  ; In tail position wrt to the if expression.
                  "bye" ; In tail position wrt to the if expression.
                  )))

; Use next to get the results one at a time:
(next) ; Use "hi".
(next) ; Use "bye" instead.
(next) ; No more choices.
; Use done? to detect the end:
(done? (next))
(done? (next))

(stage
 
 ; Display either "hi" ..., or *later* come back and display "bye" ...
 (-< (displayln "hi")
     (displayln "bye"))
 
 ; If -< was a function, the second display would be in the continuation of the first,
 ;  because of eager left-to-right argument evaluation:
 #;(define a0 (println "hi"))
 #;(define a1 (println "bye"))
 #;(-< a0 a1)
 
 ; ... continuing on and displaying this each time:
 (displayln "how's it going?"))

(next) ; Display "hi", and display "how's it going?"
(next) ; Go back and display "bye" instead, and display "how's it going?".

; Square either 1 or 2.
(stage (sqr (-< 1 2)))
(next) ; 1
(next) ; 4

(stage (define r0 (-< 1 2))
       (sqr r0))
(next) ; 1
(next) ; 4

; Square either 30 or 40, throw that away, return either 1 or 2.
(stage (sqr (-< 30 40))
       (sqr (-< 1 2)))
(next) ; Throw away 30, return 1.
(next) ;                Return 4.
(next) ; Throw away 40, return 1.
(next) ;                Return 4.

(stage (+ (-< 3 4)
          ; This *is* in the continuation of the previous expression:
          (-< 50 60)))
(next) ; 53
(next) ; 63
(next) ; 54
(next) ; 64

; That's like this:
(stage (define r1 (-< 3 4))
       (define r2 (-< 50 60))
       (+ r1 r2))

; And this:
(stage (define r0 (-< 'a 'b))
       (define r1 (-< 'c 'd))
       (list r0 r1))
(next) ; '(a c)
(next) ; '(a d)
(next) ; '(b c)
(next) ; '(b d)

; But unlike this.
; Either return 'a or return 'b, or return 'c or return 'd.
(stage (-< (-< 'a 'b)
           (-< 'c 'd)))
(next) ; 'a  idea is pick first arg of outer -<, i.e. (-< 'a 'b), then pick first arg of inner -<
(next) ; 'b
(next) ; 'c
(next) ; 'd

; Choose 'a or 'b, choose 'c or 'd, *then* return one of those.
; Unlike either of the previous ones.
(stage (define r0 (-< 'a 'b))
       (define r1 (-< 'c 'd))
       (-< r0 r1))
(next) ; 'a
(next) ; 'c
(next) ; 'a
(next) ; 'd
(next) ; 'b
(next) ; 'c
(next) ; 'b
(next) ; 'd

(stage (list (-< 'a 'b)
             (-< 'c 'd)))
(next) ; '(a c)
(next) ; '(a d)
(next) ; '(b c)
(next) ; '(b d)

(stage (-< (-< 1 2)
           (-< 3 (fail) 4)
           (fail)
           (-< 5 6)))
(next) ; 1
(next) ; 2
(next) ; 3
(next) ; 4
(next) ; 5
(next) ; 6

(stage (list (-< 1 2)
             (-< 3 (fail) 4)
             (fail)
             (-< 5 6)))
(next) ; 'done38516

(stage (list (-< 1 2)
             (-< 3 (fail) 4)
             (-< 5 6)))
(next)
(next)
(next)
(next)
(next)
(next)
(next)
(next)
#| 
'(1 3 5)
'(1 3 6)
'(1 4 5)
'(1 4 6)
'(2 3 5)
'(2 3 6)
'(2 4 5)
'(2 4 6)|#

(define (an-atom v)
  (cond [(empty? v) (fail)]
        [(list? v) (-< (an-atom (first v))
                       (an-atom (rest v)))]
        [else v]))

(stage (an-atom '(a ((b) c) d)))
(next) ; 'a
(next) ; 'b
(next) ; 'c
(next) ; 'd

(trace #false)

; To get all results as a list:
#;(list-results e
                ...)

(list-results (an-atom '(a ((b) c) d)))

(define (a-subsequence l)
  (if (empty? l)
      l
      (-< (a-subsequence (rest l))
          (list* (first l) (a-subsequence (rest l))))))

(list-results (a-subsequence '(⚀ ⚁ ⚂ ⚃ ⚄ ⚅)))
; how does it work
