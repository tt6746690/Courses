#lang racket

#| The foundation of programming and programming language
    Early 1900 foundation forr logic and mathematics
   First-order logic
      \neg \forall \land
   Set theory

   Next: the constructive subset
     i.e. algorithms, i.e. step-wise computation

   Various foundations in 1930s
      Church's Lambda Calculus
         Functions: how *we* were expressing algorithms to *each other*
      Turing's Machine
         Bits, linear array of bits, finite number of cpu states
      Goedel's recursive functions in number theory
      Post rewrite systems.

   Start with very little, notice and remove redundancies
     and then quickly build up from Church's lambda calculus
|#


#| With only functions, how do we implement booleans?
  What functionality do booleans give us?
  Choosing!

  lets make the ternary conditional.
|#


#; (define (True t e) t)
#; (define (False t e) e)

; Eager! 
#; (define (If c t e)
  (c t e))


#; (If True 123 456)  ;123
#; (If False 123 456) ; ? : , consequent if condition else alternative


#;(define (Not b)
    (If b False True))

#; (Not (Not True)) ; True

; "..." means Kleene-*

#; (define (<function-identifier> <parameter-identifier>...)
     <body-expression>)          ; specify function
#;(<identifier> <expression>...) ; function call
#;<identifier>                   ; variable access


#|  Currying: binary, ternary, etc functions via unary functions |#

; Addition in racket's standard library isnt curried
; We can make a curried one!
#; (define (⊕ x)     ; unary function that makes and return a unary function
     #;(local [(define (... y)
                 (+ x y))]
         #;...)
     (λ (y)
       (+ x y)))
; Note this function's functionality is irrelevant to the function name

; equivalent to (λ (y) (+ 100 y))
#; (⊕ 100)         ; y -> 100 + y
#; ((⊕ 100) 23)    ; 123


#; (map (⊕ 100) (list 3 2 4))
#;(map (λ (y) (+ 100 y))
     (list 3 2 4))
#;(list ((λ (y) (+ 100 y)) 3)
        ((λ (y) (+ 100 y)) 2)
        ((λ (y) (+ 100 y)) 4))


; Back to boolean

(local [(define (True t) (λ (e) t))
        (define (False t) (λ (e) e))

        ; Gives 1 unary function -> 1 unary function
        (define (If c)
          (λ (t)
            (λ (e)
              ((c t) e))))
        (define (Note b)
          (((If b) False) True))]
  
  (((If True) 123) 456)
  (((If False) 123) 456))

; implement with currying
(local [(define True (λ (t) (λ (e) t)))
        (define False (λ (t) (λ (e) e)))
        ; Gives 1 unary function -> 1 unary function
        (define If (λ (c)
                     (λ (t)
                       (λ (e)
                         ((c t) e)))))
        (define Not (λ (b)
                      (((If b) False) True)))]
  (((If True) 123) 456)
  (((If False) 123) 456))
; True is makes constant functions
; give a value, and will always return that value when invoked

; False makes identity function
; will always ignore the first argument supplied and give back the argument
; suppplied when functino invoked later


#; (define <identifier> <expression>)
#; (λ (<identifier>) <expression>)
#; (<f-expression> <a-expression>)
#; <identifier>



; r+ = r∘r*


; Agree on the convention, by former we really mean the latter
#; (Lambda (id-0 id ...) e)  ; e=expression
#; (Lambda (id-0)
           (Lambda (id ...) e))
#; (f e0 e ...)
#; ((f e0) e ...)


#; (define-syntax Lambda
     (syntax-rules ()
       [(Lambda (id-0 id ...) e)
        (Lambda (id-0)
                (Lambda (id ...) e))]
       [(Lambda (id) e)
        (λ (id) e)]))

; implement with currying
(require (only-in racket (#%app racket:app)))
(local [(define-syntax Lambda
          (syntax-rules ()
            [(Lambda (id) e) (λ (id) e)]
            [(Lambda (id-0 id ...) e)
             (Lambda (id-0)
                     (Lambda (id ...) e))]
            ))
        (define-syntax #%app
          (syntax-rules ()
            [(#%app f e) (racket:app f e)]
            [(#%app f e-0 e ...) (#%app (#%app f e-0) e ...)]))
        (define True (Lambda (t e) t))
        (define False (Lambda (t e) e))
        (define If (Lambda (c t e)
                           ((c t) e)))
        ; Gives 1 unary function -> 1 unary function
        (define Not (Lambda (b)
                            (((If b) False) True)))]
  (If True 123 456)
  (If False 123 456)
  (Not (Not True))
  ((Not True) False Not))

; #%app invisible infront of all racket
; syntactical - simply rearranging the code during compile time
;  Now function only uses unary function calls


(require (only-in racket (#%app racket:app)))
(local [; Compiler. (with only unary function)
        (define-syntax-rule (#%top . v) `v)
        (define-syntax Lambda
          (syntax-rules ()
            [(Lambda (id) e) `(λ (id) ,e)]
            [(Lambda (id-0 id ...) e)
             (Lambda (id-0)
                     (Lambda (id ...) e))]
            ))
        (define-syntax #%app
          (syntax-rules ()
            [(#%app f e) `(racket:app ,f ,e)]
            [(#%app f e-0 e ...) (#%app (#%app f e-0) e ...)]))
        ; User code.
        (define True (Lambda (t e) t))
        (define False (Lambda (t e) e))
        (define If (Lambda (c t e)
                           ((c t) e)))
        ; Gives 1 unary function -> 1 unary function
        (define Not (Lambda (b)
                            (((If b) False) True)))]
  (If True 123 456)
  (If False 123 456)
  (Not (Not True))
  ((Not True) False Not))



; Make Number!

(require (only-in racket (#%app racket:app)))
; Compiler. (with only unary function)
        #; (define-syntax-rule (#%top . v) `v)
(define-syntax Lambda
  (syntax-rules ()
    [(Lambda (id) e) (λ (id) e)]
    [(Lambda (id-0 id ...) e)
     (Lambda (id-0)
             (Lambda (id ...) e))]
    ))
(define-syntax #%app
  (syntax-rules ()
    [(#%app f e) (racket:app f e)]
    [(#%app f e-0 e ...) (#%app (#%app f e-0) e ...)]))

; User code.
(define True (Lambda (t e) t))
(define False (Lambda (t e) e))
(define If (Lambda (c t e)
                   ((c t) e)))
; Gives 1 unary function -> 1 unary function
(define Not (Lambda (b)
                    (((If b) False) True)))
; Number
(define Zero (Lambda (f x) x))
(define One (Lambda (f x) (f x)))
(define ⊕ (Lambda (m n f x) (m f (n f x))))
(define Add1 (Lambda (n) (⊕ n One)))
(define Two (Add1 One)) ; idea is to call f m+n times

; Idea is if n(False) run zero time, then True otherwise always False (constant function)
(define Zero? (Lambda (n) (n (Lambda (x) False) True)))
(define Pair (Lambda (a b selector?)
                     (If selector? a b)))
(define First (Lambda (pair) (pair True)))
(define Second (Lambda (pair) (pair False)))



(Two sqr 3)
; 81

(Zero sqr 3)
; 3

#;(⊕ One One)
#; (Lambda (f x) (f (f x)))


((Pair 123 456) True)   ; 123
((Pair 123 456) False)  ; 456
(First (Pair 123 456))  ; 123

; subtraction
; (0 0) 
; (0 1)
; (1 2)
; (2 3)
; (3 4)

(define Sub1 (Lambda (n) (First (n (Lambda (pair)   ; recursion
                                           (Pair (Second pair)
                                                 (Add1 (Second pair))))
                                   (Pair Zero Zero)))))

(Sub1 One)





