#lang racket #| CSC 324 - 2017 Fall - Lab Week 7 |#

#| Streams

 Streams are a datatype for iterable data.

 As an abstract datatype they have only three list-like or iterator-like operations:
   Empty? : stream → boolean
    First : stream → any
     Rest : stream → stream

 Many languages have stream libraries, for example Java 8 has streams meant to be used in
  a functional programming style along with Java 8's new support of anonymous functions.
  Here's the Java SDK API summary for Package java.util.stream:
    "Classes to support functional-style operations on streams of elements, such as
      map-reduce transformations on collections."
  Here's their tutorial:
   https://docs.oracle.com/javase/tutorial/collections/streams/
   
 In this lab you'll play with a general implementation of streams as lazy lists. |#

#| Lazy Lists

 A lazy list is built on-demand, delaying the evaluation of its elements.
 As a small simplification, we'll delay only the evaluation of the tail of the list,
  i.e. the result of Rest, but not the current element, i.e. the result of First.
  This will still demonstrate the main significance of the lazy evaluation.

 A lazy list will either be:
   Empty, represented by the empty list.
   A pair containing:
     • the first element
     • a function to call to produce the rest of the list |#

; Constructor.
(define-syntax-rule (List* e LL) (list e (λ () LL)))

; API.
(define Empty? empty?)
(define First first)
(define (Rest LL) ((second LL)))

; ★ Write out the compile-time expansion of each of these expressions:

(List* 1 '())
#; (list 1 (λ () '()))


(List* 1 (List* 2 '()))
#; (list 1 (λ () (list 2 (λ () '()))))


(define LL.0 (List* 1 (List* 2 (List* (/ 1 0) '()))))
#; (list 1 (λ ()
             (list 2 (λ ()
                       (list (/ 1 0) (λ () '()))))))


; ★ Predict the result of each of these expressions.
; For λs, write out the λ expression instead of racket's black-boxed "#<procedure: ...>".
; Algebraically trace as many intermediate steps as you find helpful, but include at least each pair
;  of steps showing a call of (λ () <expression>).

#;(First LL.0)
#; (first (list 1 (λ ()
                    (list 2 (λ ()
                              (list (/ 1 0) (λ () '())))))))
#; 1


(define LL.1 (Rest LL.0))
#; (define LL.1 (Rest (list 1 (λ ()
                                (list 2 (λ ()
                                          (list (/ 1 0) (λ () '()))))))))
#; (define LL.1 ((second (list 1 (λ ()
                                   (list 2 (λ ()
                                             (list (/ 1 0) (λ () '())))))))))
#; (define LL.1 ((λ ()
                   (list 2 (λ ()
                             (list (/ 1 0) (λ () '())))))))
#; (define LL.1 (list 2 (λ ()
                          (list (/ 1 0) (λ () '())))))


#;(First LL.1)
#; (first (list 2 (λ ()
                    (list (/ 1 0) (λ () '())))))
#; 2


#;(Rest LL.1)
#; ((second (list 2 (λ ()
                      (list (/ 1 0) (λ () '()))))))
#; ((λ ()
      (list (/ 1 0) (λ () '()))))
#; (list (/ 1 0) (λ () '()))
; / division by zero


#| Infinite Lists

 Lazy lists allow easily building datastructures that behave like infinite lists. |#

(define 324s (List* 324 324s))
; ★ Write out the compile-time expansion of that definition.
#; (list 324 (λ () 324s))

; ★ Predict the result of the following expression.
; Write out the intermediate steps, but hide the steps in the body of Rest.
#;(First (Rest (Rest 324s)))
#; (First (Rest (Rest (list 324 (λ () 324s)))))
#; (First (Rest ((second (list 324 (λ () 324s))))))
#; (First (Rest ((λ () 324s))))
#; (First (Rest (list 324 (λ () 324s))))
#; (First (list 324 (λ () 324s)))
#; (first (list 324 (λ () 324s)))
#; 324


; That behaves like an infinite list of 324s.

#| Core Functional Programming Operations, on Streams

 Most of the common functional programming operations on lists generalize to streams. |#

(define (Map f LL)
  (cond [(Empty? LL) LL]
        [else (List* (f (First LL)) (Map f (Rest LL)))]))

; ★ Write out the compile-time expansion of that definition.
#; (define (Map f LL)
     (cond [(empty? LL) LL]
           [else (list (f (first LL)) (λ () (Map f ((second LL)))))]))


(define N (List* 0 (Map add1 N)))
; ★ Write out the compile-time expansion of that definition.
#; (define N (list 0 (λ () (Map add1 N))))
; probably not expanding any further  


; ★ Predict the result of each of these expressions.
; Write out the intermediate steps, but hide the steps in the body of Rest.

(define N.1 (Rest N))
#; (define N.1 ((second (list 0 (λ () (Map add1 N))))))
#; (define N.1 (Map add1 N))
#; (define N.1 (cond [(empty? N) N]
                     [else (list (add1 (first N)) (λ () (Map add1 ((second N)))))]))
#; (define N.1 (list (add1 (first N)) (λ () (Map add1 ((second N))))))
#; (define N.1 (list 1 (λ () (Map add1 ((second N))))))

(define N.2 (Rest N.1))
#; (define N.2 ((second (list 1 (λ () (Map add1 ((second N))))))))
#; (define N.2 (Map add1 ((second N))))
#; (define N.2 (Map add1 (Map add1 N)))
#; (define N.2 (Map add1 (list 1 (λ () (Map add1 ((second N)))))))
#; (define N.2 (list (add1 1) (λ () (Map f ((second (list 1 (λ () (Map add1 ((second N)))))))))))
#; (define N.2 (list 2 (λ () (Map f ((second (list 1 (λ () (Map add1 ((second N)))))))))))


#;(First N.2)
#; 2

