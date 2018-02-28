#lang racket #| Macros and Runtime Defining Language M0 |#

(provide T:*id* T:*datum*
         T:set! T:if
         T:λ T:*app*
         T:block
         T:let T:local
         T:cond T:when T:while
         T:breakable T:continuable T:returnable
         T:and T:or
         Ts ; List of all the transformations, defined after all of them.
         standard-library
         M0→L0)

(require "A2.L0.rkt")

; Compile an M0 expression to an L0 expression.
(define (M0→L0 e)
  (expand (standard-library e) Ts))

#| Language M0
   ===========

 M0 is really a language and standard library: the language is essentially an extension of L0,
  that macro-compiles to L0, along with a small standard library written in that language.

 M0 is meant to be for humans to write programs in, so we won't tag it.

 There are seventeen kinds of expression, corresponding to the provides of the form ‘T:<id>’ above.
 For each of those, except T:*id*, T:*datum*, and T:*app*, there's an M0 expression (<id> <part> ...).

 Function application is then any other expression of the form: (<part> ...).
   During expansion, the macro system turns that into (*app* <id>), for T:*app* to transform.

 A simple identifier is an M0 expression meaning variable access.
   During expansion, the macro system turns that into (*id* <id>), for T:*id* to transform.

 An integer is an M0 expression meaning a constant.
  During expansion, the macro system turns that into (*datum* <id>), for T:*datum* to transform.

 It's assumed that M0 programmers will not use identifiers surrounded by asterisks. |#

; An M0 expression:
#;(λ (f a b)
    (set! a b)
    (f a 488))
; During expansion, when some parts of that are expanded, they'll be treated as if that was:
#;(λ (f a b)
    (set! a (*id* b))
    (*app* (*id* f) (*id* a) (*datum* 488)))

#| Compiling M0 to L0
   ==================

 Implement:

   1. The transformers mentioned in the ‘provide’ above, for the forms described below.

   2. The standard library, which is a function that attaches the library to an expression,
       described at the end of this file.

 Each transformer ‘T:<id>’ transforms code of the form ‘(<id> <e> ...)’.

 In the given patterns:
   ‘...’  means zero or more of the previous component
   ‘...+’ means one  or more of the previous component

 Other than T:*id*, T:*datum*, T:set!, T:*if*, T:*λ*, and T:*app*, transformers should not
  directly produce L0 forms.

 New identifiers introduced by transformations
 ---------------------------------------------
 In some of your templates you will find that you need to make up the name of an identifier.

 For new “dummy” identifiers that aren't referenced, use ‘_’.
   • we'll assume that user code does not reference any identifier with that name either

 For temporary identifiers that are referenced, use an identifier of the form "*<id>*",
  i.e. surrounded by asterisks. But don't use the names ‘*app*’, ‘*id*’, nor ‘*datum*’.
   • it will be assumed that user code does not use such identifiers |#

(module+ test (require rackunit))

; *id* *datum* set! if
; --------------------
; The following two special cases should expand to integers 0 and 1:
#;(*id* false)
#;(*id* true)
; Otherwise, they are straightforward untagged versions of L0 expressions.
; Transform those directly to their L0 form.

(define-transformer T:*id* *id*
  [e e])
(define-transformer T:*datum* *datum*
  [e e])
(define-transformer T:set! set!
  [e e])
(define-transformer T:if if
  [e e])

; λ
; -
; Extends L0's λ by:
;   allowing more than one body expression
;   allowing zero parameters, and more than one parameter
;
; Transform within the M0 language by wrapping the body in a ‘block’,
;  adding a dummy parameter if there are zero parameters,
;  and currying if there are two or more parameters.
; Transform the unary single-body-expression form to the L0 form.

(define-transformer T:λ λ
  [e e])


; *app*
; -----
; Extends L0's app by allowing zero arguments, or more than one argument.
; Transform the form with more than one argument into its curried equivalent.
; Transform the no-argument form into a one-argument form with a dummy argument [see ‘block’].
; Transform the unary form to the L0 form.

(define-transformer T:*app* *app*
  [e e])


; block
; -----
#;(block <e>
         ...)
; A sequence of zero or more expressions to be evaluated in order,
;  producing the value of the last expression,
;  or the integer 0 if there are none.
;
; Transform the form with no expressions to the integer 0.
; Transform the form with one expression to just the expression.
; Transform the form with more than one expression to a ‘let’ naming the first expression
;  with a dummy variable.
;
; For other M0 forms that need dummy values [e.g. as mentioned for *app*], use (block) for
;  the dummy value.

(define-transformer T:block block
  [e e])


; let
; ---
#;(let ([<id> <init>]
        ...+)
    <body>
    ...+)
; Evaluates the <init>s in order, then introduces the distinctly named local variables <id>s,
;  initialized by the values of the <init>s, then evaluates the <body>s as a block.
;
; Transform using the standard LC transformation: to an expression that makes and immediately calls
;  a function.

(define-transformer T:let let
  [e e])


; local
; -----
#;(local [(define (<f-id> (<id> ...))
            <f-body>
            ...+)
          ...+]
    <body>
    ...+)
; Introduces the distinctly named local <f-id>s into scope, to functions created in that scope,
;  then evaluates the <body>s as a block.
;
; Transform using the standard LC+set! transformation: to an expression that initializes
;  all the <f-id>s to dummy values, sets them to their functions, then evaluates the body.

(define-transformer T:local local
  [e e])


; and or
; ------
#;(and <e0> <e> ...+)
#;(or  <e0> <e> ...+)
; Standard short-circuiting operators for two or more boolean expressions.
; Transform to ‘if’s or ‘cond’s.

(define-transformer T:and and
  [e e])
(define-transformer T:or or
  [e e])


; cond
; ----
#;(cond [<condition> <result>
                     ...+]
        ...+)
#;(cond [<condition> <result>
                     ...+]
        ...
        [else <else-result>
              ...+])
; Evaluates the boolean <condition>s in order, until the first true one or possibly the else,
;  then evaluates the corresponding <result>s as a block.
;
; Transform using ‘if’s, ‘when’s, and/or ‘block’s.

(define-transformer T:cond cond
  [e e])


; when
; ----
#;(when <condition>
    <body>
    ...+)
; If boolean <condition> is true evaluates the <body>s as a block, otherwise produces a dummy value.

(define-transformer T:when when
  [e e])


; while
; -----
#;(while <condition>
         <body>
         ...+)
; A standard while loop.
; Transform to a recursive no-argument function that is immediately called.

(define-transformer T:while while
  [e e])


; returnable breakable continuable
; --------------------------------
#;(returnable <e>
              ...+)
#;(breakable <e>
             ...+)
#;(continuable <e>
               ...+)
; Evaluates the <e>s as a block, in a local scope containing the identifier ‘return’,
;  ‘break’, or ‘continue’ bound to the continuation that escapes the entire expression.
; These are meant to be used manually by the programmer: around a function body, loop, or loop body,
;  to return early, break, or continue.

(define-transformer T:returnable returnable
  [e e])
(define-transformer T:breakable breakable
  [e e])
(define-transformer T:continuable continuable
  [e e])


; List of all the transformations.
(define Ts (list T:*id* T:*datum*
                 T:set! T:if
                 T:λ T:*app*
                 T:block
                 T:let T:local
                 T:cond T:when T:while
                 T:breakable T:continuable T:returnable
                 T:and T:or))

; Standard Library
; ----------------
; Add definitions for the functions described by the comments in the body.
(define (standard-library e)
  `(local [
           ; Boolean logic
           ; -------------
           ; (not b) : the negation of b, implemented with ‘if’
           
           ; Arithmetic
           ; ----------
           ; (- a b) : the difference between a and b
           (define (- a b) (+ a (⊖ b)))
           ; (⊖ a) : the negative of a
           ; (> a b) : whether a is greater than b
           ; (>= a b) : whether a is greater than or equal to b
           ; (= a b) : whether a is equal to b
           ]
     ,e))
