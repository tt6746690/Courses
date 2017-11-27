 #lang racket #| Capturing Control Flow |#

(require (only-in racket/control prompt abort call/comp))

#| Call-with-Composable-Continuation |#

#;(call/comp unary-f) ; Pass this expression's continuation, up to most recent prompt, to ‘unary-f’.
                      ; the expression between prompt and call/comp is captured and assigned with name unary-f

; A little wrapper for that:
(define-syntax-rule (let-continuation id ; Name this whole expression's continuation as ‘id’.
                                      ; Evaluate the body expressions in order.
                                      ; These are *not* part of the continuation.
                                      ; Use the last expression for the value of the whole expression.
                                      body
                                      ...) ; The spot *after* is named ‘id’.
  (call/comp (λ (id) body ...)))

; Nothing exciting, since we don't use the continuation.
(prompt (+ (+ 1 20)
           (let-continuation k
                             (+ 300 4000))))
; 4321

; Somewhere to save continuations.
(define K (void))

; Still evaluates to 4321, but as a side-effect stores a continuation.
(prompt (+ (+ 1 20)
           (let-continuation k
                             (set! K k)
                             (+ 300 4000))))
(K 50000)
#;(<procedure:+> 21
                 50000)

(prompt (+ (local []
             (displayln '(+ 1 20))
             (+ 1 20))
           (let-continuation k
                             (displayln '(set! K k))
                             (set! K k)
                             (+ 300 4000))
           ; k is this point in the control flow: *after* the ‘let-continuation’.
           ))
; (+ 1 20)
; (set! K k)
; 4321

(K 50000)   ; note (+ 1 20) is computed in the continuation
; 50021

(local []
  (prompt (let-continuation k
                            (set! K k)
                            (println 'a)
                            (abort)
                            (println 'b)) ; ‘k’ is from here ...
          (println 'c)) ; ... to here.
  (println 'd))
; 'a
; 'd

(K)
; 'c
; why not print d?
; because k is a slice of function of the remaining exprs inside a prompt
