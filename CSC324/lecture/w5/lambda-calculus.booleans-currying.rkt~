#lang racket ; Choose "The Racket language" from the Language dialog in DrRacket.

#| The Foundation of Programming and Programming Languages.

 Early 1900s: foundation for logic and mathematics.

    First-order Logic.
      ¬, ∨, ∧, ⇒, ⇐, ∃, ∀, and predicates.
    Set Theory [typically ZFC].
      Some axioms and an axiom schema, expressed in first-order logic.

 Next, the "constructive" subset, i.e. algorithms, i.e. step-wise computation.

   Various foundations in the 1930s.
     Church's Lambda Calculus.
       Functions: essence of how we were expressing algorithms to each other.
     Turing's Machine.
       Bits, linear memory with only sequential access, cpu finite state machine:
        essence of how digital machines would behave.

 We'll start with very little, notice and remove redundancies to use even less, then
  rapidly build up from a very small core to modern programming language features. |#


#| With only functions, how do we represent other data?

 Let's start with the simplest datatype: boolean.

 What functionality do booleans give us? The ability to choose.
 Let's make a ternary conditional: |#

#;(If True  123 456) ; 123
#;(If False 123 456) ; 456

; Make booleans make the choice:
#;(define (False then else) else)
#;(define (True  then else) then)
#;(True  123 456) ; 123
#;(False 123 456) ; 456

#;(define (If condition consequent alternative)
    ; Nothing we write here can give us short-circuiting, so we'll have to revisit
    ;  "If as a function" later.
    (condition consequent alternative))

; We'll build natural numbers in the next lecture, so for now let's play with booleans:
#;(define (Not b)
    (If b False True))

#;(Not True)
#;(Not False)
#;(Not (Not False))


#| Our language of <expression>s so far is defined by structural induction/recursion: |#

; 1. Variable reference [a base case].
#;<identifier>

; 2. Numeric literal [a base case].
#;<numeric-literal>

; 3. Function creation + naming [a recursive case].
#;(define (<identifier> <identifier> ...)
    <body-expression>)

; 4. Function call [a recursive case].
#;(<identifier> <expression> ...)


#| Currying.

 Binary, ternary, etc, functions via unary functions. |#

; Addition in racket's standard library isn't curried, but we can make a curried interface.
(define (⊕ x) ; Unary function.
  (λ (y) ; Producing a unary function.
    (+ x y)))

(⊕ 100)
; Algebraic substitution of value into the body:
(λ (y)
  (+ 100 y))

((⊕ 100) 23)
; Substitute:
((λ (y)
   (+ 100 y))
 23)
; Substitute:
(+ 100 23)

(map (⊕ 100) (range 1 23 4))
(map (λ (y) (+ 100 y)) (list 1 5 9 13 17 21))
(list ((λ (y) (+ 100 y))  1)
      ((λ (y) (+ 100 y))  5)
      ((λ (y) (+ 100 y))  9)
      ((λ (y) (+ 100 y)) 13)
      ((λ (y) (+ 100 y)) 17)
      ((λ (y) (+ 100 y)) 21))
(list (+ 100  1)
      (+ 100  5)
      (+ 100  9)
      (+ 100 13)
      (+ 100 17)
      (+ 100 21))
(list 101 105 109 113 117 121)

#;{(define (True  then) (λ (else) then))
   (define (False then) (λ (else) else))
   ((True  123) 456) ; 123
   ((False 123) 456) ; 456
   (define (If condition)
     (λ (consequent)
       (λ (alternative)
         ((condition consequent) alternative))))
   (((If False) 123) 456) ; 456
   (((If True)  123) 456) ; 123
   (define (Not b) (((If b) False) True))
   (Not True)
   (Not False)
   (Not (Not False))}

; Revised set of expressions:
#;<variable-identifier>
#;<numeric-literal>
#;(λ (<parameter-identifier>) <body-expression>)
#;(<function-expression> <argument-expression>)
#;(define (<function-identifier> <parameter-identifier>)
    <body-expression>)

; Revised to remove some redundancy:
#;<variable-identifier>
#;<numeric-literal>
#;(λ (<parameter-identifier>) <body-expression>)
#;(<function-expression> <argument-expression>)
#;(define <identifier> <expression>)

(define True  (λ (then) (λ (else) then)))
(define False (λ (then) (λ (else) else)))
((True  123) 456) ; 123
((False 123) 456) ; 456
(define If (λ (condition)
             (λ (consequent)
               (λ (alternative)
                 ((condition consequent) alternative)))))
(((If False) 123) 456) ; 456
(((If True)  123) 456) ; 123
(define Not (λ (b) (((If b) False) True)))
(Not True)
(Not False)
(Not (Not False))
