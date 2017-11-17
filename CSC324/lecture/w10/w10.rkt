#lang racket #| state without state |#
#|
  without state cannot be automatically translated to and from state
|#

; Let's express the problem of number the leaves of a full binary tree
;  First using state, then not

; we'll use 2-element lists for non-leaf parts of binary tree.

(define (counter)
  (define the-count 0)
  (λ ()
    (set! the-count (add1 the-count))
    (sub1 the-count)))


#; (define (number-leaves count bt)
     (cond [(list? bt) (list (number-leaves count (first bt))
                             (number-leaves count (second bt)))]
           [else (count)]))


#; (number-leaves (counter) '(a ((b c) d)))
; '(0 ((1 2) 3))


#; (define (number-leaves count bt)
     (cond [(list? bt) (define count-and-result-0
                         (number-leaves count (first bt)))
                       (define count-and-result-1
                         (number-leaves (first count-and-result-0) (second bt)))
                       (list (first count-and-result-1)
                             (list (second count-and-result-0)
                                   (second count-and-result-1)))]
           [else (list (add1 count) count)]))

;(number-leaves 0 'a) '(1 0)
;(number-leaves 1 '((b c) d))
;(number-leaves 1 '(b c))
;(number-leaves 1 'b) '(2 1)
;(number-leaves 2 'c) '(3 2)
;(number-leaves 3 'd) '(4 3)
;(number-leaves 0 '(a ((b c) d))) '(4 (0 ((1 2) 3)))

(match-define x 123)
(match-define (list a b) (list 12 345))

#; (define (get count) (list count count))
#; (define (put count new-count) (list new-count (void)))
#; (define (return count r) (list count r))

#; (define (number-leaves count bt)
     (cond [(list? bt)
            (match-define (list count-0 result-0) (number-leaves count (first bt)))
            (match-define (list count-1 result-1) (number-leaves count-0 (second bt)))
            (return count-1 (list result-0 result-1))]
           [else (match-define (list count-0 c) (get count))
                 (match-define (list count-1 _) (put count-0 (add1 c)))
                 (return count-1 c)]))


#; (define-syntax Do
     (syntax-rules (←)
       [(Do state
            (id ← (f  a ...))
            clause ...
            r)
        (match (f state a ...)
          [(list state′ id) (Do state′       ; nested ...
                                clause ...
                                r)])]
       [(Do state
            r)
        (return state r)]))

; function return a pair (state, result)
; state is hidden inside syntax rules, i.e. state-0, state-1
; while result is named i.e. result-0 result-1
#; (define (number-leaves count bt)
     (cond [(list? bt)
            (Do count  ; monad
                (result-0 ← (number-leaves (first bt)))
                (result-1 ← (number-leaves (second bt)))
                (list result-0 result-1))]
           [else
            (Do count
                (c ← (get))
                (_ ← (put (add1 c)))
                c)]))

; monad
; a sequence of computation (in chunks)
; subsequent computation may depend on previous result
; or a computation that combines a result

#; (define (eval e)
     (cond [(number? e) e]
           [else (define r0 (eval (first e)))
                 (cond [(equal? r0 #false) #false]
                       [else (define r1 (eval (third e)))
                             (cond [(equal? r1 #false) #false]
                                   [else (cond [(equal? (not (zero? r1)) #false) #false]
                                               [else (/ r0 r1)])])])]))


(define-syntax Do
  (syntax-rules (←)
    [(Do (id ← e)
         clause ...
         r)
     (local [(define id e)]
       (cond [(equal? id #false) #false]
             [else (Do clause
                       ...
                       r)]))]
    [(Do r)
     r]))

(define (eval e)
  (cond [(number? e) e]
        [else (Do (r0 ← (eval (first e)))
                  (r1 ← (eval (third e)))
                  (_  ← (not (zero? r1)))
                  (/ r0 r1))]))




(require rackunit)
(check-equal? (eval '((1 + (5 + 2)) + (2 + 3))) 3/5)
(check-equal? (eval '((1 + (0 + 2)) + (2 + 3))) #false)



; monad is for bundling computation on states

; what if programming language does not have macro system
; just wrap in lambdas that takes in state, put return value of previous computation
; as state argument to next lambda and call them sequentially
#; (λ (count) (number-leaves count (first bt)))
#; (λ (count) (number-leaves count (second bt)))
#; (λ (count) (return count (list result-0 result-1)))

; or ...
#; (fix-2nd number-leaves (first bt))
#; (fix-2nd number-leaves (second bt))
#; (fix-2nd return (list result-0 result-1))



; function exclusively operate on states, they take in state and return states
#; (define (number-leaves bt)
     (λ (count)
       (cond [(list? bt)
              (match-define (list count-0 result-0)
                ((number-leaves (first bt)) count))
              (match-define (list count-1 result-1)
                ((number-leaves (second bt)) count))
              (return (list result-0 result-1))]
             [else (match-define (list count-0 c) (get count))
                   (match-define (list count-1 _) (put count-0 (add1 c)))
                   (return count-1 c)])))

(define (>>= f g)
  (λ (count)
    (match-define (list c r) (f count))
    ((g r) c)))

(define (get count) (λ (count) (list count count)))
(define (put count new-count) (λ (count) (list new-count (void))))
(define (return count r) (λ (count) (list count r)))

(define (number-leaves bt)
  (λ (count)
    (cond [(list? bt)
           (>>= (number-leaves (first bt))
                (λ (r0) (>>= (number-leaves (second bt))
                           (λ (r1) (return (list r0 r1))))))]
          [else (>>= (get)
                     (λ (c)
                       (>>= (put (add1 c)))
                       (λ (_) (return c))))])))























