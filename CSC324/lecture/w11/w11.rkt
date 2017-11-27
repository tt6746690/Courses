#lang racket


(define ((f x) y)
  (+ x y))

(define ((return v) count) (list v count))
(define ((put new-count) count) (list (void) new-count))
(define ((get) count) (list count count))


; bind operator
(define ((>>= a b) count)
  (match-define (list r c) (a count))
  ((b r) c))


(define ((>> a b) count)
  (match-define (list r c) (a count))
  (b c))


; post-increment
(define ++ (>>= (get)
                (λ (r)
                  (>> (put (add1 r))
                      (return r)))))



; returns a function waiting for a count
(define (number-leaves bt)
  (cond [(list? bt)
         (>>= (number-leaves (first bt))
              (λ (result-0)
                (>>= (number-leaves (second bt))
                     (λ (result-1)
                       (return (list result-0 result-1))))))]
        [else ++]))

((number-leaves '(a ((b c) d))) 0)
((number-leaves 'a) 0)


#; ((>> (put 123)
        (get))
    456)










