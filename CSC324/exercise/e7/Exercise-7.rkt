#lang racket #| CSC324 2017 Fall Exercise 7 |#

#| In this exercise you implement path finding via backtracking, expressing functions producing
    multiple results in single-result style.

 Both yielding and backtracking illustrate “inversion of control”, where the caller of a function
  takes control of the flow. This allows the big-bang animation at the end of the file to animate
  the paths algorithm that is reimplemented here.

 From the amb library, use only -< and fail in your implementation.
 Produce values only on-demand. In particluar, do not create a list and then pick elements from it
  [e.g. using an-element].
 Factor out repeated parts of a computation, especially for -<. For example, write (sqr (-< 2 3))
  rather than (-< (sqr 2) (sqr 3)). |#

(require "amb.rkt")
(trace #false)

; Implement backtracking function adjacent to produce one of the four points adjacent to spot.
; Use map appropriately.
; Using identity and reverse can make the symmetry of the choices especially explicit.
(define (adjacent spot)
  (local [(define (add-coord s1 s2)
            (list (+ (first s1) (first s2)) (+ (second s1) (second s2))))]
    (add-coord (-< '(0 1) '(0 -1) '(1 0) '(-1 0))
               spot)))

; (x y)
; (x+1 y) (x-1 y) (x y+1) (x y-1)

; Implement assert to take a boolean, and fail unless it's true.
(define (assert b)
  (if b b (fail)))

; Implement assert-then to take a condition expresson and one or more body expressions,
;  producing an expression that:
;   1. Fails unless the condition is true.
;   2. Otherwise evaluates the body expressions in order, producing the value of the last one.
(define-syntax-rule (then-assert c expr ...)
  (cond [(equal? c #t) expr ...]
        [else (fail)]))


(define (without element a-list)
  (filter (λ (x) (not (equal? x element))) a-list))

(define (include-in-all element list-of-lists)
  (map (λ (a-list) (list* element a-list))) list-of-lists)


(define (not-in? element a-list)
  (equal? (without element a-list) a-list))

(define (in? element a-list)
  (not (not-in? element a-list)))

; Implement backtracking function path to produce a path from from to to through through.
; If partial? is true, include each valid partial path.
; Use assert and/or then-assert appropriately.
#;(define (path to through from [partial? #true])
    (list (then-assert (equal? to from)
                       (path to (without from through) (adjacent from)))))

#;(define (path to through from [partial? #true])
    (local [(define adj (adjacent from))
            (define valid-adj
              (cond [(equal? (member adj through) #f) (fail)]
                    [else (first (member adj through))]))
            (define path-from-adj (path to (without from through) valid-adj))]
      (list (without from through))))


(define (path to through from [partial? #true])
  (define (next-path to through from p)
    (local []
      (assert (in? from through))
      (-< (then-assert partial? p)
          (cond [(equal? from to) (append p (list from))]
                [else (next-path to (without from through) (adjacent from) (append p (list from)))])))
    )
  (next-path to through from '()))


; The idea
; Let backtracking explore all paths,
; use assert to stop the backtrack whenever the coordinates not in `through`
; use then-assert to stop the backtrack whenever we have found the path, i.e. to=from




#;(define (paths to through from)
    (include-in-all
     from
     (cond [(not-in? from through) (list)]
           [(equal? to from) (list (list))]
           [else (local [(define (a-p new-from)
                           (paths to (without from through) new-from))]
                   (apply append (map a-p (adjacent from))))])))



;   (include-in-all
;    from
;    (cond [(not-in? from through) (list)]
;          [(equal? to from) (list (list))]
;          [else (apply append (map (fix-1st-2nd paths to (without from through)) (adjacent from)))])))
  
(require (except-in rackunit fail))
(define figure-eight (list (list 0 0) (list 1 0) (list 2 0)  ; list of coordinates
                           (list 0 1)            (list 2 1)
                           (list 0 2) (list 1 2) (list 2 2)
                           (list 0 3)            (list 2 3)
                           (list 0 4) (list 1 4) (list 2 4)))

(check-equal? (without '(1 0) '((1 0) (0 0))) '((0 0)))
(check-equal? (list-results (path (list 2 4) figure-eight (list 0 0)))
              '(()
                ((0 0))
                ((0 0) (0 1))
                ((0 0) (0 1) (0 2))
                ((0 0) (0 1) (0 2) (0 3))
                ((0 0) (0 1) (0 2) (0 3) (0 4))
                ((0 0) (0 1) (0 2) (0 3) (0 4) (1 4))
                ((0 0) (0 1) (0 2) (0 3) (0 4) (1 4) (2 4))
                ((0 0) (0 1) (0 2))
                ((0 0) (0 1) (0 2) (1 2))
                ((0 0) (0 1) (0 2) (1 2) (2 2))
                ((0 0) (0 1) (0 2) (1 2) (2 2) (2 3))
                ((0 0) (0 1) (0 2) (1 2) (2 2) (2 3) (2 4))
                ((0 0) (0 1) (0 2) (1 2) (2 2))
                ((0 0) (0 1) (0 2) (1 2) (2 2) (2 1))
                ((0 0) (0 1) (0 2) (1 2) (2 2) (2 1) (2 0))
                ((0 0))
                ((0 0) (1 0))
                ((0 0) (1 0) (2 0))
                ((0 0) (1 0) (2 0) (2 1))
                ((0 0) (1 0) (2 0) (2 1) (2 2))
                ((0 0) (1 0) (2 0) (2 1) (2 2) (2 3))
                ((0 0) (1 0) (2 0) (2 1) (2 2) (2 3) (2 4))
                ((0 0) (1 0) (2 0) (2 1) (2 2))
                ((0 0) (1 0) (2 0) (2 1) (2 2) (1 2))
                ((0 0) (1 0) (2 0) (2 1) (2 2) (1 2) (0 2))
                ((0 0) (1 0) (2 0) (2 1) (2 2) (1 2) (0 2) (0 3))
                ((0 0) (1 0) (2 0) (2 1) (2 2) (1 2) (0 2) (0 3) (0 4))
                ((0 0) (1 0) (2 0) (2 1) (2 2) (1 2) (0 2) (0 3) (0 4) (1 4))
                ((0 0) (1 0) (2 0) (2 1) (2 2) (1 2) (0 2) (0 3) (0 4) (1 4) (2 4))
                ((0 0) (1 0) (2 0) (2 1) (2 2) (1 2) (0 2))))


; The following animates the path finding algorithm.
(require 2htdp/image 2htdp/universe)
(define (animate to through from)
  (define background (rectangle (+ 3 (apply max (map first  through)))
                                (+ 3 (apply max (map second through)))
                                "solid" "black"))
  (define (spots-image spots colour background)
    (match spots
      ['() background]
      [`((,x ,y) . ,spots) (spots-image spots colour
                                        (place-image/align (square 1 "solid" colour)
                                                           (add1 x) (add1 y) "left" "top"
                                                           background))]))
  (define building (spots-image through "white" background))
  
  (stage (path to through from))
  (big-bang (next)
            [to-draw (λ (path) (scale 25 (spots-image path "blue" building)))]
            [on-tick (λ (_) (next)) 1/2]
            [stop-when done?]))

(animate (list 2 4) figure-eight (list 0 0))
