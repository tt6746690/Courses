#lang racket #| Type Check the Simply Typed LC [‘STLC’, aka ‘λ→’] by Symbolic Evaluation |#

#| Simply Typed Lambda Calculus

  Reference for the STLC, for students wishing to go further:
    Pierce, Benjamin C. “Types and programming languages”.
      Available online via your UofT library account.

  Types:
    some base types [λ→ can have any set of base type, all considered disjoint]
    a type constructor: (type → type)
    base type false, to represent lack of a valid type [the “bottom” type, aka ‘⊥’]

  Some notable type aspects λ→ lacks: subtyping, polymorphism.

  Expressions:
    type
    numeric-literal
    string-literal
    identifier
    (λ (identifier type) expression)
    (expression expression)

  The following overrides evaluation so that evaluating an expression type checks it, and produces the
   type of the expression. If an expression does not type check its type is #false.

  We imagine this as part of a compiler.
  In other words this would ‘run’ the program at *compile time*, to *just* type check it.
  Types will be *run time* values during this *compile time* type checking evaluation of the program.

  A value is now a type, which is a “symbolic” representative of any value of that type.
  That core meaning of “symbolic” in static analysis.

  All numbers will be represented by one representative value: the symbol 'Number.
  Similarly, 'String stands for a string.
  Unary functions will be represented by lists of the proper form.

  In particular, evaluation below is overriden so that the following evaluate as indicated:
    numeric-literal : 'Number
     string-literal : 'String
    (λ (id type) expr) : `(,type → ,type-of-expression), if expression type checks
    (f-expr a-expr) : result-type, if f-expr type checks as `(,argument-type → ,result-type),
                       and a-expr type checks as argument-type.

  Evaluation of identifiers, with the proper scoping, is automatic without overriding the default.

  Also, we will use (define id expr), also without overriding. |#

; The implementation is *optional* material.
; Read the comments, then skip to the usage that follows (require 'λ→).
(module λ→ racket
  (provide (rename-out [datum #%datum]
                       [app #%app]
                       [λ→ λ] [λ→ lambda]))

  (require (for-syntax syntax/parse))
  ; Override numeric and string literals to produce the symbols 'Number and 'String.
  (define-syntax datum (syntax-parser [(_ . datum:number) #''Number]
                                      [(_ . datum:string) #''String]
                                      [(_ . datum) #''datum]))

  ; Override unary function call to “call” '(A → R) with 'A to produce 'R, otherwise produce #false.
  (define-syntax app (syntax-parser [(_ f a) #'(match (list f a)
                                                 [(list `(,a′ → ,r) a′) (and a′ r)]
                                                 [_ #false])]))

  ; Override unary λ to require parameter type annotation, and immediately evaluate the body
  ;  in the scope of the parameter with that type.
  (define-syntax λ→
    (syntax-parser [(_ (an-id:id a-type:expr) body:expr)
                    ; Use built-in λ for naming, to maximize the connection to “ordinary” evaluation.
                    #'`(,a-type → ,((λ (an-id) body) a-type))])))
(require 'λ→)

; All numbers at runtime [if we stick to the STLC subset] are the one symbolic number 'Number:
123
324

"hello"
"bye"

; These are considered the two atomic values, and so we should be able to use them directly:
'Number
'String

; We can also name a/the number:
(define n 456)
n

; Technically these are not in our subset, but illustrate general typing of function call:
('(X → Y) 'X)
('(X → Y) 'Z)

; The one function, representing all such functions, taking a number and producing a string:
'(Number → String)
; Call a/the representative of such functions, on a representative of a number or string:
('(Number → String) 'Number)
('(Number → String) 123)
('(Number → String) "hi")

; Name a function:
(define string-length '(String → Number))
; This is being run only for type checking, so that is literally the value of ‘string-length’.
string-length
(string-length "abc")
(define abc "abc")
(string-length abc)
(string-length 123)

(define add1 '(Number → Number))
(define sub1 add1) ; For type checking, these two functions are literally the same.

(add1 (string-length abc))
(string-length (add1 abc))
(add1 (sub1 (string-length abc)))

(define string-append '(String → (String → String)))

(string-append "a")
((string-append "a") abc)
(string-length ((string-append "a") abc))
(add1 (string-length ((string-append "a") abc)))

(λ (s 'String) 0)
((λ (s 'String) 0) "abc")
((λ (s 'String) 0) 123)

; Type checking starts with the idea of evaluating every expression once, with a simplified semantics.
;
; Regardless of when and how many times this will be called [zero, more than one], type checking
; evaluates it once, immediately, with the representative of all valid arguments. Just as values
; are representative of any/all value(s) of a certain type, calling a function creation expression
; once represents, mininally but generally, its behaviour for all of its calls.
(λ (s 'String) (add1 (string-length ((string-append "a") s))))

(define f (λ (s 'String) (add1 (string-length ((string-append "a") s)))))

(f "abc")
(f 123)
; The distinction between the function call operation, which combines a function and an argument,
;  and passing the argument to the function, is especially important here.
; The two previous expressions perform function call, but do *not* evaluate the body of the function.
; In particular, as usual for eager by-value call, the function is *not* in control of the evaluation.

(define * '(Number → (Number → Number)))
(define cube (λ (x 'Number) ((* ((* x) x)) x)))
cube
(cube 123)
(cube abc)
