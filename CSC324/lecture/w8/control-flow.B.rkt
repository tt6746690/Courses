#lang racket #| Delimiting Control Flow, and Aborting |#

#| Two primitives.

 • ‘prompt’ : delimit some control flow for ‘abort’, evaluate body expressions/definitions in order,
               produce the result of the last expression
 • ‘abort’  : abort the current control flow, jumping to the end of the most recent ‘prompt’ that
               control flow passed through, insert the argument [if any] into the continuation |#

(require (only-in racket/control abort prompt))

#;(abort) ; Aborts the program.

; Prints 'a and 'b, then aborts the program.
#;(local []
    (println 'a)
    (println 'b)
    (abort)
    (println 'c))

; Prints 'a and 'b, jumps to end of inner prompt, prints 'd, jumps to end of outer prompt, prints 'f.
; 'a, 'b, 'd, 'f
(prompt (println 'a)
        (prompt (println 'b)
                (abort)
                (println 'c))
        (println 'd)
        (abort)
        (println 'e))
(println 'f)

(+ (+ 1 20)
   ; Aborts and places 50000 into the continuation of the whole prompt expression.
   (prompt (abort 50000)
           (+ 300 4000)))
; i.e. reduce to
#; (+ 21 50000)

; a is defined to be 20 by abort
(define (a)
  ; Abort out to the “return” step, returning 20.
  (prompt (+ 1 (abort 20) 300)))

(+ 4000 (a))

(define (b) (+ 1 (abort 20) 300))

; Which prompt is aborted to is dynamic, like exception handling: it's the last *executed* prompt.
(+ 4000 (prompt (b)))
; 4020  (note prompt is not defined inside b)

(prompt (+ 4000 (b)))
; 20    (because abort directly gives control flow back to toplevel repl)



; None of these abort the program:
(prompt (abort))           ; abort caught by a prompt
(define (g) (abort))       ; delayed evaluation since in lambda
(prompt (g))               ; no-op, 
(define (c) (prompt (g)))  ; no-op
(c)


(prompt (+ 1
           (local []
             (abort 20)
             30)
           400))
; 20

(+ 1
   (prompt (abort 20)
           30)
   400)
; 421

; A prompt expression
#;(prompt body ...)
;  is like making a function
#;(define (p) body ...)
;  and calling it, with abort inside ‘p’ meaning ‘return’

(local []
  (println 'A)
  (println (sqr (prompt (println 'B)
                        (abort 18)
                        (println 'C))))
  (println 'D))
; 'A
; 'B
; 324
; 'D

#;(local []
    (println 'A)
    (println (sqr (p)))
    (println 'D))
#;(define (p)     ; think of prompt as a function then calling it
    (println 'A)
    (return 18)
    (println 'B))
