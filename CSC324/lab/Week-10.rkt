#lang racket #| CSC324 2017 Fall Week 10 Lab. |#

#| 1. Implement the functions ‘an-element’, ‘pair-of’, ‘with’, and ‘shuffled’, using -< and fail.
   2. Implement ‘match/fail’.
   3. Re-implement ‘an-element’ and ‘with’ using ‘match/fail’.
   4. Implement ‘prime-factor’. |#

(require "amb.rkt") ; Library from Nov 3rd.
#;(trace #false)

; An alternative to list-results, with less noise during tracing, and usable like ‘next’.
(define (nexts)
  (define n (next))
  (if (done? n)
      '()
      (list* n (nexts))))

(module+ test (require rackunit)
  
  (stage (an-element '(a b c d)))
  (check-equal? (next) 'a)
  (check-equal? (next) 'b)
  (check-equal? (nexts) '(c d))

  (stage (pair-of '(a b c) '(d e)))
  (check-equal? (nexts) '((a d) (a e) (b d) (b e) (c d) (c e)))

  (stage (with 'd '(a b c)))
  (check-equal? (nexts) '((d a b c) (a d b c) (a b d c) (a b c d)))

  (stage (shuffled '(a b c)))
  (check-equal? (nexts) '((a b c) (b a c) (b c a) (a c b) (c a b) (c b a))))

#| An element of a-list. |#
(define (an-element a-list)
  (cond [(empty? a-list) (fail)]
        [(equal? (rest a-list) '()) (-< (first a-list))]
        [else (-< (first a-list) (an-element (rest a-list)))]))

#| A two-element list with its first element coming from list A, and second element from list B. |#
(define (pair-of A B)
  (list (an-element A) (an-element B)))

#| The list a-list with e inserted somewhere. |#
(define (with e a-list)
  (cond [(empty? a-list) (list e)]  ; (list e) in place for (rest a-list) where a-list is empty
        [else (-< (list* e a-list)  ; idea is either e inserted as first element in a-list or not
                  (list* (first a-list) (with e (rest a-list))))]))

#| A list of the elements of a-list, in some order.
   Hint: use ‘with’. |#
(define (shuffled a-list)
  (-< (with (first a-list) (rest a-list))  ; idea is inserting first element to the rest
      (with (first a-list) (reverse (rest a-list)))))

#| Like match, but fail if no match. |#
(define-syntax-rule (match/fail expr()
                                clause
                                ...)
  (match expr
    clause
    ...))

#| Patterns to match first and rest of a list. |#
(module+ test
  
  (check-equal? (match '(a b c) [(list* first′ rest′) (list first′ rest′)])
                (list 'a '(b c)))
  (check-equal? (match '(a b c) [`(,first′ . ,rest′) (list first′ rest′)])
                (list 'a '(b c)))
  
  ; Those two kinds of patterns are also valid expressions to create a list from a first and rest.
  (check-equal? (list* 'a '(b c)) '(a b c))
  ; In quotation, the ‘.’ is like an infix ‘list*’.
  (check-equal? '(a . (b c)) '(a b c))
  (check-equal? `(,(+ 1 2) . ,(map sqr '(3 4))) '(3 9 16))
  
  ; Patterns can also use ‘...’:
  (check-equal? (match '(a b c) [`(,first′ ,rest′ ...) (list first′ rest′)])
                (list 'a '(b c)))

  ; tests
  (stage (an-element′ '(a b c d)))
  (check-equal? (next) 'a)
  (check-equal? (next) 'b)
  (check-equal? (nexts) '(c d))

  (stage (with′ 'd '(a b c)))
  (check-equal? (nexts) '((d a b c) (a d b c) (a b d c) (a b c d)))

  ; prime factor
  (stage (prime-factor 14))
  (check-equal? (next) 2)
  (check-equal? (nexts) '(3 5 7 11 13))
  )


(define (an-element′ a-list)
  (match a-list
    ['() (fail)]
    [`(,x) (-< x)]
    [`(,x ,rest ...) (-< x (an-element′ rest))]))


(define (with′ e a-list)
  (match a-list
    ['() '(e)]
    [`(,x ,rest ...) (-< `(,e . ,a-list)
                         `(,x . ,(with e rest)))]))


(require math/number-theory) ; the cheat....
#| A prime factor of positive natural number n, at least as large as d, including repetitions.
   Pre-condition: all prime factors of n are at least as large as d. |#
(define (prime-factor n [d 2]) ; [d 2] here means: ‘d’ is an optional argument, defaulting to 2.
  (an-element′ (filter prime? (range d n))))



