#lang racket


(require
  (only-in racket/control prompt abort call/comp))

(define-syntax-rule (let-continuation id body ...)
  (call/comp (λ (id) body ...)))

(define alternates '())
(define-syntax -<
  (syntax-rules ()
    [(-< e0 e1)
     (let-continuation choice-point
                       (display "Putting aside: ")
                       (println 'e1)
                       (define alternate
                         (λ ()                             ; function that can be called again
                           (display "Retrying with: ")
                           (println 'e1)
                           (choice-point e1)))             ; puts value of e1 at choice-point
                       (set! alternates (list* alternate
                                               alternates))
                       (display "Using: ")
                       (println 'e0)
                       e0)]
    [(-< e0 e1 e2 ... ) (-< e0 (-< e1 e2 ...))]))

#;(local []                    ; stack is textually nested
    (-< 6 7)
    (define r (random 100))
    (displayln r)
    (+ 1 (-<  (-< (* 2 10)     ; continuation of this is inner -< is outer -< is + 
                  (* 3 100))    
              (-< (* 4 10)
                  (* 5 100))))
    )

(displayln "After block . . .")

#; ((first alternates))


(define (next)
  (when  (empty? alternates) (abort))
  (define alternate (first alternates))   ; stack like behavior
  (set! alternates (rest alternates))
  (alternate))

; a control flow path that is a deadend
(define (fail)
  (abort (next)))


; second arg is in continuation of first arg
#;
(list (prompt (+ (-< 1 2)        ; prompt: any saving of continuation is up to prompt
                 (-< 30 40))))
#;
(list (-< 1 2) (-< 3 4))

#; 
(define (an-atom v)
  (cond [(empty? v) (fail)]
        [(list? v) (-< (an-atom (first v))
                       (an-atom (rest v)))]
        [else v]))

; -< ambiguous choice operator
; for backtracking
; prolog designed for ML, only has backtracking control flow
; when computer gets more powerful, dont use it any longer

; kanren


(define (an-atom v)
  (cond [(empty? v) (abort (next))]
        [(list? v) (-< (an-atom (first v))
                       (an-atom (rest v)))]
        [else v]))



; thinking in indiviaul result, no need for apply append map
(define (a-subsequence l)
  (cond [(empty? l) l]
        [else (-< (a-subsequence (rest l))
                  (list* (first l) (a-subsequence (rest l))))]))

; use backtracking to consturct list of subsequences of a list
#;
(prompt (define results '())
        (prompt (define s (a-subsequence '(3 2 4)))   ; added prompt here to prevent going back to outer prompt
                (set! results (list* s results))
                (fail))
        results)

(define-syntax-rule (list-all e)
  (local [(define results '())]
    (prompt (set! results (list* e results))
            (fail))
    results))

; why have to use define-syntax-rule instead of define
; 

#; 
(list-all (an-atom '(a (b c (d) (e)) g)))

#;
(list-all (a-subsequence '(a b c)))

#;
(list-all ((-< sqr - sqrt)
           (-< 1 2 3 (fail) 4 5)))
           ; not a list, conceptually just try each of the control flow ... 

#; ;args evaluated, second arg aborts arg evaluation control flows 
(list 1 (fail) 2)

#; ; saying we pick one of the expression and evaluate
(-< 1 (fail) 2)

#;
(local []
  (println (-< 1 2))
  (fail)
  (println (-< 3 4)))











