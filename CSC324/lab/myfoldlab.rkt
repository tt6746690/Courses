#lang racket

(require (except-in rackunit fail))


; INTRODUCING FOLD

; in racket, every base function that /can/ be variadic likely already is.
; but let's say we only had the binary forms of some classic variadic functions:


(define (my-add a b) (+ a b))
(define (my-prod a b) (* a b))
(define (my-string-append a b) (string-append a b))

; how can we use higher-order programming to recover the variadic versions
; in a systematic way? below you will implement my-fold, an operation which
; takes a binary function and a list and 'folds' that function over the list


; let's begin more concretely:
; (uncomment the check-equal?s as you implement each function)

; first, define var-add to return the sum of a list of numbers
; use my-add from above. do not use + !!!
(define (var-add ls)
  (if (empty? ls)
      (void)   ; ★ replace (void) with something
      (void))) ; ★ replace (void) with something

; second, define var-prod to return the product of a list of numbers
; use my-add from above. do not use + !!!
(define (var-prod ls)
  (if (empty? ls)
      (void)   ; ★ replace (void) with something
      (void))) ; ★ replace (void) with something

#;(check-equal? (var-add '(1 2 3 4)) 10)
#;(check-equal? (var-prod '(1 2 3 4)) 10)



; abtracting what you've learned above, implement my-fold
; which takes a binary operation (bin-op) and a 'base' value
; and returns a function which can 'fold' that operation over a list
(define ((my-fold bin-op base) ls)
  (if (empty? ls)
      (void) ; ★ replace (void) with something
      (void))) ; ★ replace (void) with something

#;(check-equal? ((my-fold my-add 0) '(1 2 3 4)) 10)
#;(check-equal? ((my-fold my-prod 1) '(1 2 3 4)) '24)
#;(check-equal? ((my-fold my-string-append "") '("a" "b" "c" "d")) "abcd")



; now let's get slightly fancier.

; cons is the constuctor for the list type. it takes two
; arguments, a value and a list, and prepends that value onto the list
; note that (cons a b) is the same as (list* a b)

; implement 'my-list', which does the same thing as 'list'
; using only my-fold and cons
; note that . below allows us to define a variadic function
; whose arguments are captured as a list called 'ls'

(define (my-list . ls)
  (void)) ; ★ replace (void) with something


#;(check-equal? (my-list 1 2 3 4) (list 1 2 3 4))



; now implement 'my-map', which does the same thing as 'map',
; using only my-fold and cons (and lambda)

; if you have trouble writing in directly, try to seperately define the
; function which you're going to fold over:

; apply-and-prepend should take a function f, and return a binary function
; which  takes a value and a list and returns the result of applying that
; function to the value,  and the prepending the returned value to a list
; (see the check-expect)

; you should only have to use cons and function application
(define ((apply-and-prepend f) element ls)
  (void)) ; ★ replace (void) with something

#;(check-equal? ((apply-and-prepend sqr) 2 '(1 1 1)) '(4 1 1 1))
#;(check-equal? ((apply-and-prepend not) #t '(1 1 1)) '(#f 1 1 1))

; now use only cons and my-fold and lambda
; (or only my-fold and apply-and-prepend)
; to write my-map:
(define (my-map f ls)
  (void))  ; ★ replace (void) with something


#;(check-equal? (my-map sqr '(1 2 3 4)) '(1 4 9 16))



; note that there are many variations on fold.
; we've implemented right fold, called foldr in racket
; (in many languages, foldr is simply called 'fold')

; sometimes we may want to fold from the left as well (foldl).
; for associative operations like my-add and my-prod these are
; going to behave identically, but for non-associative operations
; like subtraction, foldr and foldl are different:

(foldr - 0 '(1 2 3 4))
(foldl - 0 '(1 2 3 4))

; (note that the racket syntax is a bit different than i used above)

; now use foldl and cons to implement my-reverse, which does the same thing as reverse
(define (my-reverse ls)
  (void))  ; ★ replace (void) with something

#;(check-equal? (my-reverse '(1 2 3 4)) '(4 3 2 1))

; also for some situations it doesn't make sense to fold over an
; empty list, or maybe not a list with a single element either
; in such a case we might have a version of fold which doesn't take a base
; and simply produces an error if the list is too short

; finally, just as racket allows us to map over multiple lists
; we can make a variadic fold which takes n lists, and an function
; which takes n+1 values

; that's it!








