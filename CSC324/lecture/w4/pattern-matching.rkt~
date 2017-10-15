#lang racket #| Quasiquote and Pattern Matching on Runtime Data. |#

#| Let's make a little language of Calculus functions.

 The expressions in our language are defined by Structural Induction/Recursion:
   • cos
   • sin
   • (<f> + <g>) , where <f> and <g> are expressions
   • (<f> · <g>) , where <f> and <g> are expressions
   • (- <f>)     , where <f> is an expression

 We'll represent ‘cos’, ‘sin’, ‘+’, ‘·’, and ‘-’ by symbols with those names,
  and parenthesized forms by lists of three or two elements accordingly. |#

; cos:
(quote cos)
'cos

; sin:
(quote sin)
'sin

; It's just a coincidence that two global variables in racket have names ‘cos’ and ‘sin’:
cos
sin

(define f 'cos)
(define g 'sin)

; One of the inductive rules says that (f + g) is now in the language.
; But not literally:
'(f + g)
; Instead, it means this is in the language:
(list f '+ g)
'(cos + sin)

#| Quasiquotation.

 Syntax:

   `<form>
   (quasiquote <form>)

 The '`' is a back-tick.

 Quasiquotation is like quote, except that parts of the value can be non-literal, i.e.
  computed. The computed parts are ones that are unquoted:

   ,<form>
   (unquote <form>)

 You might be familiar with the concept from templating in web programming. |#

`(,f + ,g)
(quasiquote ((unquote f)
             +
             (unquote g)))
(list (quasiquote (unquote f))
      (quasiquote +)
      (quasiquote (unquote g)))
(list f
      (quote +)
      g)
(list (quote cos)
      (quote +)
      (quote sin))

; See "Quasiquote and Unquote" in:
;   http://www.ccs.neu.edu/home/matthias/HtDP2e/Draft/i2-3.html

; Some more terms in our language:
'((cos + sin) · sin)
'(- ((cos + sin) · sin))

#| Pattern matching.

 Values can be de-structured with the ‘match’ operation.

 It takes an expression for a value, followed by clauses of the form:
   [<pattern> <result-expression>]

 The square-brackets are a convention for the human reader, in particular emphasizing that
  the grouping doesn't mean call <pattern> on the <result-expression>.

 The value is compared with each pattern, in order, and if the variables in the pattern
  could be assigned values to create the value being matched, the clause defines those
  variables locally and uses the corresponding <result-expression> as its result value. |#

; A function for differentiation, defined by Structural Recursion.
(define (δ f)
  (match f

    ; Literals can be used as patterns.
    ['cos '(- sin)]
    ['sin 'cos]

    ; The list constructor can be used in patterns.
    [(list f '+ g) (list (δ f) '+ (δ g))]
    ; Quasiquote can be used in patterns.
    #;[`(,f + ,g) `(,(δ f) + ,(δ g))]
    
    [`(- ,f) `(- ,(δ f))]

    [`(,f · ,g) `((,(δ f) · ,g)
                  +
                  (,(δ g) · ,f))]
    ; Alternative in terms of the list constructor.
    #;[(list f '· g) (list (list (δ f) '· g)
                           '+
                           (list (δ g) '· ,f))]))

; Trace of a call:
(δ '(cos + sin))
(δ (list 'cos '+ 'sin))
(local [(define f 'cos)
        (define g 'sin)]
  (list (δ f) '+ (δ g)))
(list '(- sin) '+ 'cos)
; Also written as:
'((- sin) + cos)

; Third derivative of cos · sin:
(δ (δ (δ '(cos · sin))))
