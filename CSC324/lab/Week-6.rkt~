#lang racket ; CSC 324 - 2017 Fall - Lab Week 6

#| Now that we're in #lang racket, there are expressions with side-effects.


 Variables can be "mutated": changed after their initial definition.
 The purpose of a mutation is not its immediate value, instead the purpose is the effect on the
  evaluation of later expressions.

 Anything detectable besides the value that an expression produces is called a "side-effect",
  or sometimes in programming language work just an "effect". Sometimes, expressions having
  a side-effect are called "statements".

 In racket the operation [not function!] ‘set!’ updates a variable: (set! <identifier> <expression>).


 The Interactions area can be "mutated": I/O can be explicitly performed.

 DrRacket was already printing the values of top-level expressions automatically.
 To print a value explicitly there is the unary function ‘println’ [and many others].


 Without side-effects, it is pointless to evaluate a sequence of expressions.
 Now that we have side-effecting statements, there are places where a sequence of expressions is
  allowed. The statements are evaluated in order, with the value of the last one's values used as
  the value of the sequence.

 Idea is order matters for statements (i.e. with side-effects) only


 In particular, ‘cond’ can
   • have a sequence of statements for the result in a clause
   • lack an ‘else’ clause, which can be thought of as "else don't cause any side-effects"

 Function and local bodies can also be a sequence of statements. |#


#| Some examples of the above. |#
(define v 1000)
(cond [(zero? (random 2))
       ; I/O isn't done for its value.
       (println "Changing v.")
       ; A pointless expression to have in the middle of a sequence.
       (sqr v)
       ; Mutation isn't done for its value.
       (set! v (random 1000))])
v ; Implicit println.

#| For the special-case of a single-branch conditional, there are ‘when’ and ‘unless’, which then
    don't need the parenthesization of the condition-result. |#

(when (zero? (random 2))
  (println "Changing v.")
  (set! v (random 1000)))
v

(println "Probably will change.")
(unless (zero? (random 1000))
  (set! v (random 1000)))
v


#| ★ Implement while loops.

 A particular while loop is implemented below with a local expression that creates and calls
  a recursive function. Replace "replace me" with that implementation, and then identify
  which part is the condition, and which are the statements, and replace those with the
  names ‘condition’ and ‘statement’. |#

(define-syntax while
  (syntax-rules ()
    [(while condition
            statement
            ...)
     (local [(define (loop)
               (when condition statement ... (loop)))]
       (loop))]))


(local [(define x 3210)
        (define y 456)]
  (while (not (= x y))
         (println (max x y))
         (cond [(< x y) (set! y (- y x))]
               [else    (set! x (- x y))]))
  x)


;   ????
; (local [(define x 3210)
;         (define y 456)]
;   ; A while loop is a tail-recursive function that, when the condition is true,
;   ;  evaluates the body statements and repeats by calling itself.
;   (local [(define (loop)
;             (when (not (= x y))
;               (println (max x y))
;               (cond [(< x y) (set! y (- y x))]
;                     [else    (set! x (- x y))])
;               (loop)))]
;     ; Start the loop.
;     (loop))
;   x)


#| ★ Implement list comprehensions.

 Fix the first ‘check-equal?’ expression.
 Then look at the second one for an intended form of usage for the list comprehension ‘List’.

 Implement ‘List’, by showing how you took the element-expr (+ 1 (* 2 x)), the identifier x,
  and the list-expr '(3 2 4), and inserted them into an expression that produces '(7 5 9). |#

(require rackunit)

(check-equal? '(7 5 9)
              ; Replace the following with an expression that contains the expressions
              ;  '(3 2 4) and (+ 1 (* 2 x)), to produce '(7 5 9).
              (map (λ (x) (+ 1 (* 2 x))) '(3 2 4)))

(define-syntax List
  (syntax-rules (∈ :)
    [(List element-expr : id ∈ list-expr)
     (map (λ (id) element-expr) list-expr)]))

(check-equal? '(7 5 9)
              (List (+ 1 (* 2 x)) : x ∈ '(3 2 4)))


#| ★ Implement for-in loops.

 Write code that doesn't use ‘for’, to mimic the intended behaviour of the example usage below.

 Possible approaches:
   1. A recursive function that takes the remainder of the list as an argument.
   2. A while loop that updates a variable containing the remainder of the list.
   3. Something like a list comprehension.
 
 Abstract that to implement ‘for’. |#

(define-syntax for
  (syntax-rules (∈ :)
    [(for id ∈ list-expr :
       statement
       ...)
     (map (λ (id) (cond [#t statement ...])) list-expr)]))

(define sum 0)
(for e ∈ '(3 2 4) :
  (set! sum (+ sum e))
  (println sum))
sum
