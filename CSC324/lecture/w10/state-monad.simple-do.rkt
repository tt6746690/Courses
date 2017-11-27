#lang racket #| State without state. |#

#| The Lambda Calculus, and Pure Functional programming in general, doesn't have explicit state
    [mutable variables or values].

 The lambda calculus can however compute any computable function.

 We saw some programming constructs expressed as relatively small and simple transformations.
 For example:
   (if <condition> <consequent> <alternative>)
   →
   (if <condition> (λ () <consequent>) (λ () <alternative>))
 More importantly, that's a local transformation: it doesn't require altering surrounding code.

 Also, people and tools find reasoning about the properties of ‘if’ by reasoning about that
  transformed code to be relatively easy, and to capture the important properties of ‘if’.

 For another example, here's a recent paper proposing a deep extension to Haskell's already deep
   type system: http://i.cs.hku.hk/~bruno/papers/hs2017.pdf
 Look at Figure 1 for their model of Haskell expressions, on the line labelled “term”:
   e ::= x | λx.e | e1 e2 | let x=e1 in e2
 The rest of the model is about the type system.
 It turns out that expanding any kind of Haskell expression into the lambda calculus [with let],
  and reasoning about the type of that expanded code, allows reasoning well about the original
  expression.

 Modelling state is not as straightforward. To reason about programming languages for single
  programs running on a machine, the lambda calculus is usually extended with a mutation operation,
  and modelled with a “store”: essentially ‘set!’, and the environment-closure memory model.

 Nevertheless, various stateful design patterns have been identified and captured in pure
  functional programming, and the techniques have implications for imperative programming.
  For example, search the web for: javascript async monad . |#


; Let's express the problem of numbering the leaves of a full binary tree:
;  first using state, and then without.
; We'll uses lists with two elements to represent non-leaf trees.

#;(number-leaves '(a ((b c) d)))
#;'(0 ((1 2) 3))

(define (counter)
  (define the-count 0)
  (λ ()
    (set! the-count (add1 the-count))
    (sub1 the-count)))

(define (number-leaves′ a-counter bt)
  (cond [(list? bt) (list (number-leaves′ a-counter (first bt))    ; recurse to subtrees
                          (number-leaves′ a-counter (second bt)))]
        [else (a-counter)]))  ; the-count-- returned

(require rackunit)

(check-equal? (number-leaves′ (counter) '(a ((b c) d)))
              '(0 ((1 2) 3)))


; To remove the state, we'll return it as a value along with the result, in a list.
; function returns (count, result)  --- count returned is total count
(define (number-leaves′′ count bt)
  (cond [(list? bt) (define count-and-result-0 (number-leaves′′ count
                                                                (first bt)))
                    (define count-and-result-1 (number-leaves′′ (first count-and-result-0)
                                                                (second bt)))
                    ; returns count returned from second computing number-leaves from right subtree
                    (list (first count-and-result-1)
                          ; and also the result, (left_subtree, right_subtree)
                          (list (second count-and-result-0) (second count-and-result-1)))]
        [else (list (add1 count) count)]))  ; (add1 count) is the current count after this leaf

(check-equal? (number-leaves′′ 0 '(a ((b c) d))) '(4 (0 ((1 2) 3))))
(check-equal? (number-leaves′′ 0 'a) '(1 0))
(check-equal? (number-leaves′′ 1 '((b c) d)) '(4 ((1 2) 3)))
(check-equal? (number-leaves′′ 1 '(b c)) '(3 (1 2)))
(check-equal? (number-leaves′′ 1 'b) '(2 1))
(check-equal? (number-leaves′′ 2 'c) '(3 2))
(check-equal? (number-leaves′′ 3 'd) '(4 3))


; Let's now write that very uniformly and explicitly.

; Name the bundling of state with result.
(define (return count v) (list count v))
; Name the update of state.
(define (put count new-count) (list new-count (void)))
; Name the getting of state for calculation.
(define (get count) (list count count))

; Although we'll switch to the familiar match form in the syntax rule [it has the right
;  scoping to avoid having to work out how to generate new variable names], we'll use
;  (match-define <pattern> <expression>) temporarily to clearly see the uniformity.

; (match-define (list a b) '(1 2))
; a is 1 and b is 2

(define (number-leaves′′′ count bt)
  ; Each state name is used exactly once, as the first argument in the next expression.
  (cond [(list? bt)
         (match-define (list count-0 result-0) (number-leaves′′′ count   (first bt)))
         (match-define (list count-1 result-1) (number-leaves′′′ count-0 (second bt)))
         (return count-1 (list result-0 result-1))]
        [else
         (match-define (list count-0 c) (get count))
         (match-define (list count-1 _) (put count-0 (add1 c)))
         (return count-1 c)]))

(check-equal? (number-leaves′′′ 0 '(a ((b c) d))) '(4 (0 ((1 2) 3))))


; Now we make a simple “Monad Do Notation” for the “state monad”, threading an initial
;  state through a sequence of expressions.
(define-syntax Do
  (syntax-rules (←)
    [(Do count              ; count is the extra state information, added to argument/return statement
         (id ← (f a ...))   ; id available for clauses below, result of computation
         clause ...
         r)
     (match (f count a ...) ; takes count as first arg, returns a (count, id) pair
       [(list count′ id) (Do count′     ; the returned state count′ as input to the next procedure in the sequence 
                             clause ...
                             r)])]
    [(Do count
         r)    ; r is in tail position, is the return value of the monad
     (return count r)]))



; Now we can write:
(define (number-leaves count bt)
  (cond [(list? bt) (Do count
                        (result-0 ← (number-leaves (first bt)))
                        (result-1 ← (number-leaves (second bt)))
                        (list result-0 result-1))]
        [else (Do count
                  (c ← (get))            ; get is injected with count, (get count) called
                  (_ ← (put (add1 c)))   ; put is injected with count, (put count (add1 c) called
                  c)]))

(check-equal? (number-leaves 0 '(a ((b c) d))) '(4 (0 ((1 2) 3))))

; Exercise: add another clause to Do for expressions whose result value isn't used.
; For example, allow the use of put shown below:
#;(define (number-leaves count bt)
    (cond [(list? bt) (Do count
                          (result-0 ← (number-leaves (first bt)))
                          (result-1 ← (number-leaves (second bt)))
                          (list result-0 result-1))]
          [else (Do count
                    (c ← (get))
                    (put (add1 c))
                    c)]))
