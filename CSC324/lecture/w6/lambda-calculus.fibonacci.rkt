#lang racket #| Completeness of the Lambda Calculus |#

#| Syntactic Language of the Lambda Calculus.

 Let Ids be a set of identifiers. 
 To avoid textual ambiguity: identifiers can't contain spaces nor parentheses, and Lambda ∉ Ids.

 1. Each id ∈ Ids is a term.
 2. If id ∈ Ids and e is a term, then (Lambda (id) e) is a term.
 3. If e0 and e1 are terms, then (e0 e1) is a term. |#

#| Semantics of Lambda Calculus Terms.

 1. id : Variable access, which is also the default meaning of the id as an expression in racket.
 2. (Lambda (id) e) : unary closure creation, expressed in racket by the expression (λ (id) e).
 3. (e0 e1) : Function call, expressed implicitly in racket by (e0 e1), explicitly by (#%app e0 e1).

 For #3, we'll use eager by-value. |#

#| Some common notational shorthands.

 The following ‘define-syntax’es define compile-time notations, expressing various forms as
  shorthands for Lambda Calculus terms. 

 The notational forms give meaning to:
   • non-unary functions
   • non-unary function calls
   • naming a function and referring to it in its body
   • sequential local naming with a body expression in the scope of the names,
      with shorthands for function creating-and-naming
   • delaying evaluation of the consequent and alternative in a ternary conditional

 To evaluate the result in racket, there are two extra clauses that translate the two kinds of
  non-identifier Lambda Calculus terms to racket code. |#

(require (only-in racket [#%app app]))

(define-syntax Lambda
  (syntax-rules ()
    [(Lambda (id) e) (λ (id) e)] ; Lambda Calculus → Racket.
    [(Lambda (id0 id ...) e) (Lambda (id0) (Lambda (id ...) e))]
    [(Lambda () e) (Lambda (•) e)]))
        
(define-syntax #%app
  (syntax-rules ()
    [(#%app e0 e1) (app e0 e1)] ; Lambda Calculus → Racket.
    [(#%app e0 e1 e ...) (#%app (#%app e0 e1) e ...)]
    [(#%app e0) (#%app e0 (Lambda (•) •))]))

(define-syntax Rec
  (syntax-rules ()
    [(Rec (f-id id) e)
     (Local [(Define (f-id f′-id id)
                     ; In e, expressions of the form (f-id e′) are shorthand for (f-id f′-id e′).
                     (local [(define-syntax f-id
                               (syntax-rules ()
                                 [(f-id e′) (f′-id f′-id e′)]))]
                       e))]
            (Lambda (v) (f-id f-id v)))]
    [(Rec (f-id id0 id ...) e) (Rec (f-id id0) (Lambda (id ...) e))]
    [(Rec (f-id) e) (Rec (f-id •) e)]))
        
(define-syntax Local
  (syntax-rules (Define Rec) ; Require 'Define' and 'Rec' to appear literally where mentioned.
    [(Local [(Define (Rec f-id id ...) e0)
             definition
             ...]
            e1)
     (Local [(Define f-id (Rec (f-id id ...) e0))
             definition
             ...]
            e1)]
    [(Local [(Define (f-id id ...) e0)
             definition
             ...]
            e1)
     (Local [(Define f-id (Lambda (id ...) e0))
             definition
             ...]
            e1)]
    [(Local [(Define id e0)
             definition
             ...]
            e1)
     ((Lambda (id)
              (Local [definition
                       ...]
                     e1))
      e0)]
    [(Local [] e1) e1])) ; Simplest base case for sequence of local definitions.
        
(define-syntax If
  (syntax-rules ()
    [(If c t e) (c (Lambda () t) (Lambda () e))]))


; To see a Lambda Calculus number in familiar decimal notation, we'll call:
(define (decimal n) ((n add1) 0))
;
; To see the Lambda Calculus terms rather than run them, comment out that definition of ‘decimal’,
;  uncomment these definitions, and run the file again:
#;(define-syntax-rule (decimal n) n)
#;(define-syntax-rule (λ (id) e) (quasiquote (λ (id) (unquote e))))
#;(define-syntax-rule (app e0 e1) (quasiquote ((unquote e0) (unquote e1))))
#;(define-syntax-rule (#%top . id) (quote id))

; A Lambda Calculus Program to Compute the 24th Fibonacci Number.  
(Local
 [(Define (True  consequent  •) (consequent))
  (Define (False • alternative) (alternative))
  
  (Define (Zero f x) x)
  (Define (One  f x) (f x))
  (Define (⊕ m n f x) (m f (n f x)))
  (Define Add1 (⊕ One))
  (Define Two (⊕ One One))
  (Define Four (⊕ Two Two))
  (Define Eight (⊕ Four Four))
  (Define Sixteen (Four Two))
  (Define (Zero? n) (n (Lambda (•) False) True))
          
  (Define (Pair a b selector) (If selector a b))
  (Define (First  p) (p True))
  (Define (Second p) (p False))

  (Define (Sub1 n) (First (n (Lambda (pair) (Pair (Second pair) (Add1 (Second pair))))
                             (Pair Zero Zero))))

  ; This is an exponential algorithm for fibonacci.
  ; Also, we have the overhead of (Sub1 n) and (Zero? n) being Θ(n), and many other inefficiencies.
  (Define (Rec fibonacci n)
          (If (Zero? n)
              Zero
              (If (Zero? (Sub1 n))
                  One
                  (⊕ (fibonacci (Sub1 n))
                     (fibonacci (Sub1 (Sub1 n)))))))]

 ; Computers are fast!
 (decimal (fibonacci (⊕ Sixteen Eight))))
