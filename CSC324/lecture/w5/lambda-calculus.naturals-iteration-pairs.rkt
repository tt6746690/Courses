#lang racket #| Lambda Calculus [LC] : Naturals, Iteration, Pairs |#


; Import eager by-value function call again, with the name ‘app’.
(require (only-in racket [#%app app]))


; Uncomment these and run again to see everything written out in the core language of LC+Define.
; Turning into a list of symbols, or a parse tree
; representing the meaning of lambda calculus (consisting of only unary function) 
#; (define-syntax-rule (λ (id) e) `(Lambda (,id) ,e)) ;drop in expression, but dont quote it, so want it to be expanded
#; (define-syntax-rule (app e0 e1) `(,e0 ,e1))
#; (define-syntax-rule (define id e) `(Define ,id ,e))
#; (define-syntax-rule (#%top . id) 'id) ; Override [unbound] variable reference.


#| Curried notation.

 And: Unary function creation compiled to racket unary function creation.
      Unary function call compiled to racket eager by-value call. |#

(define-syntax Lambda
  (syntax-rules ()
    [(Lambda (id0 id1 id ...) e) (Lambda (id0) (Lambda (id1 id ...) e))]
    [(Lambda () e) (Lambda (·) e)]  ; lambda of no parameter
    [(Lambda (id) e) (λ (id) e)])) ; LC → Racket.

; In racket: function call terms have an implicit invisible ‘#%app’ in front, which gives us a hook
;  to rewrite them into whatever we want.
(define-syntax #%app
  (syntax-rules ()
    [(#%app e0 e1 e2 e ...) (#%app (#%app e0 e1) e2 e ...)]    ; make unary function call recursively
    [(#%app e0) (#%app e0 (Lambda (·) ·))]                     ; a variable does nothing
    [(#%app e0 e1) (app e0 e1)])) ; LC → Racket.               ; unary function call with racket's #%app


#| Function creation-and-naming notation.

 And: Naming compiled to racket naming. |#

(define-syntax Define
  (syntax-rules ()
    [(Define (f-id id ...) e) (Define f-id (Lambda (id ...) e))]   ; (define (f args...) expr) equivalent to (define f (λ (args...) expr)
    [(Define id e) (define id e)])) ; → Racket.


#| Booleans |#

(Define (True  consequent  ·) consequent)
(Define (False · alternative) alternative)
(Define (If condition consequent alternative) (condition consequent alternative))


#| Numbers

 A number will represent: compose/iterate that many times.

 E.g. (Four sqr 3) will compute (sqr (sqr (sqr (sqr 3)))).
      (Four sqr)   will compute the equivalent of (Lambda (x) (sqr (sqr (sqr (sqr x))))).
      (Four f x)   will compute (f (f (f (f x)))). |#

(Define (Zero f x)           x)  ; (Zero f) ≡ Identity
#; (Define Zero (Lambda (f x) x))
#; (Define Zero (Lambda (f) (Lambda (x) x)))
#; (define Zero (λ (f) (λ (x) x)))

(Define (One  f x)        (f x)) ; (One  f) ≡ f
(Define (Two  f x)    (f (f x))) ; (Two  f) ≡ f∘f
#; (define Two (λ (f) (λ (x) (f (f x)))))
#; (Two sqr)
#; (λ (x) (sqr (sqr x)))


(Define (Add1 n f x) (f (n f x))) ; (Add1 n f) ≡ f∘(f∘⋯∘f) where the latter is composed n times.
#; (Add1 Two f x)
#; (f (Two f x))
#; (f (f (f (x))))   ; which is Three...

(Define (⊕ m n f x) (m f (n f x))) ; (⊕ m n f x) ≡ (f∘⋯∘f)((f∘⋯∘f)(x)) with m and n compositions.

#;(⊕ One Two)
#;(Lambda (f x) ((One f) (Two f x)))              ; Note (One f) returns a function, by def is a lambda
#;(Lambda (f x) ((Lambda (x′) (f x′)) (f (f x))))
#;(Lambda (f x) (f (f (f x))))

; Note we can freely substitute in algebraic steps
; Since no mutation or side effects

(Define Four    (⊕ Two Two))
(Define Eight   (⊕ Four Four))
(Define Sixteen (⊕ Eight Eight))

#;(define (rep n f seed)
    (cond [(zero? n) seed]
          [else (rep (sub1 n) f (f seed))]))

#| for _ in range(n): seed = f(seed)
   return seed |#

#;(rep 0 f seed) #;seed
#;(rep 1 f seed) #;(f seed)
#;(rep 2 f seed) #;(f (f seed))

#;(rep 0 (λ (·) #false) #true) #;#true
#;(rep 1 (λ (·) #false) #true) #;#false
#;(rep 2 (λ (·) #false) #true) #;#false

#;(define (zero? n) (rep n (λ (·) #false) #true))
; the inner lambda outputs #false if called
; so if lambda evaluated zero times, it outputs #true otherwise #false

#| seed = True
   for _ in range(n): seed = False
   return seed|#

#;(Zero (Lambda (·) False) True) #;True
#;(One  (Lambda (·) False) True) #;False
#;(Two  (Lambda (·) False) True) #;False

(Define (Zero? n) (n (Lambda (·) False) True))

#; (Define (Pair a b selector) (If selector a b))
(Define (Pair a b) (Lambda (selector) (If selector a b)))
(Define (First  p) (p True))
(Define (Second p) (p False))

#;(Pair 123 456)
#;(Lambda (selector) (If selector 123 456))

#;(First (Lambda (selector) (If selector 123 456)))
#;((Lambda (selector) (If selector 123 456)) True)
#;(If True 123 456)
#;123

#;(Second (Pair 123 456))
#;456

#| (Sub1 n) subtracts One, except leaves Zero as Zero:
     (a, b) = (0, 0)
     for _ in range(n): (a, b) = (b, b + 1)
     return a

   (0 0)
   (0 1)
   (1 2)
   (2 3)
|#
(Define (Sub1 n) (First (n (Lambda (pair) (Pair (Second pair) (Add1 (Second pair))))
                           (Pair Zero Zero))))

; Leave the system [use racket's add1 and numeric literals] to see numbers as decimal digits.
(define (decimal N) ((N add1) 0)) #;(add1 (add1 (add1 (⋯ (add1 0)))))
(decimal Sixteen)
(decimal (Sub1 Sixteen))
