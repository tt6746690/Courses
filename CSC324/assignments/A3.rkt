#lang racket #| CSC 324 2017 Fall Assignment 3  |#

#| A Type Checker with some Inference |#

(provide type) ; Implement ‘type’.

#|
https://piazza.com/class/j7azb9w9yrb4ub?cid=320
https://piazza.com/class/j7azb9w9yrb4ub?cid=314
|#


#| The Language of Types.

 Base types: Boolean, Number, String, Void, and ⊥ [called “bottom”, \bot].
   • Represented by symbols with those names.

 Function type constructor: (<Type> ... → <Type>)
   • Represented by a list of zero or more types, followed by the symbol →, followed by a type. |#


#| The Syntactic Language of Expressions.

 <base-literal> - represented by a boolean, number, or string.

 <identifier> - represented by a symbol.

 For each of the following, a **list** with that structure, where ‘...’ means zero or more of the
  preceding component, parts in angle-brackets are from the corresponding category, and other
  non-list components appear literally as symbols. Except: <Type> is never ⊥.

 (λ ((<identifier> : <Type>) ...)
   <body-expression>
   ...
   <result-expression>)

 (let ([<identifier> <expression>] ...)
    <body-expression>
    ...
    <result-expression>)

 (rec (<function-identifier> (<identifier> : <Type>) ... : <result-Type>)
   <body-expression>
   ...
   <result-expression>)
 
 (<function-expression> <argument-expression> ...)

 (if <condition-expression>
     <consequent-expression>
     <alternative-expression>)

 (set! <identifier> <expression>) |#

#| The Type of an Expression.

 As with evaluation, the type of an expression is relative to a current environment that contains
  a mapping of **variables in scope to their types**, i.e. (variable id, type) pair
 Also, if at any point a type during the check of an expression is ⊥, the type of the whole expression
  is ⊥, and the expression is said to “not type check”.

 <base-literal> : the corresponding base type

 <identifier> : the type of the most local identifier with that name in the environment

 (λ ((<identifier> : <Type>) ...)
   <body-expression>
   ...
   <result-expression>)

   • In the current environment with each <identifier> bound locally to its corresponding <Type>:
      if each <body-expression> type checks then the type is (<Type> ... → <Result-Type>),
      where <Result-Type> is the type of <result-expression>.

 (let ([<identifier> <expression>] ...)
    <body-expression>
    ...
    <result-expression>)

   • If <expression> ... type check as <Type> ..., then the type is the same as the type of:

       ((λ ((<identitifer> : <Type>) ...)
          <body-expression>
          ...
          <result-expression>)
        <expression> ...)

 (rec (<function-identifier> (<identifier> : <Type>) ... : <result-Type>)
   <body-expression>
   ...
   <result-expression>)

   • In the current environment with <function-identifier> bound locally to <result-Type>,
      the type is the same as the type of:
       (λ ((<identitifer> : <Type>) ...)
         <body-expression>
         ...
         <result-expression>)

 (<function-expression> <argument-expression> ...)

   • Type checks iff the type of <function-expression> is a function type and the types of
      <argument-expression> ... match the function type's argument types, in which case
      the type is the function type's result type.

 (if <condition-expression>
     <consequent-expression>
     <alternative-expression>)

   • Type checks iff the type of the condition is Boolean, and the consequent and alternative
      have the same type, in which case the type of the expression is the type of the consequent.

 (set! <identifier> <expression>)

   • Type checks iff the type of the most local identifier with that name in the environment
      is the type of <expression>, in which case the type is Void. |#


; An environment is a list of 2-element lists: ((<identifier>, <type>) ...)
; where <identifier> and <type> are symbols

(define (find-type-by-id env id)
  (local [(define filtered-env
            (filter (λ (x) (equal? id (first x))) env))]
    (cond [(empty? filtered-env) '⊥]
          [else (second (first filtered-env))])))

(define (zip list-1 list-2)
  (map (λ (x y) (list x y)) list-1 list-2))

(define (zip2 list-1 list-2 sep)
  (map (λ (x y) (list x sep y)) list-1 list-2))

(define (in? element a-list)
  (not (equal? a-list (filter (λ (x) (not (equal? x element))) a-list))))

(module+ test (require rackunit)
  ; helpers
  (check-equal? (find-type-by-id '((x y) (a b)) 'x) 'y)
  (check-equal? (find-type-by-id '((x 1) (x 2)) 'x) '1)  ; finds most recent, i.e. front of list
  (check-equal? (find-type-by-id '((x y) (a b)) 'z) '⊥)  ; no identifier in env
  (check-equal? (zip '(a b) '(Number Boolean)) '((a Number) (b Boolean)))
  (check-equal? (zip2 '(1 2 3) '(a b c) ':) '((1 : a) (2 : b) (3 : c)))
  (check-equal? (in? 'a '(a b c)) #t)
  (check-equal? (in? 'x '(a b c)) #f)
  ; literal
  (check-equal? (type 11) 'Number)
  (check-equal? (type #t) 'Boolean)
  (check-equal? (type #f) 'Boolean)
  (check-equal? (type "s") 'String)
  ; symbol
  (check-equal? (type 's '((s Boolean))) 'Boolean)
  (check-equal? (type 's '((x Boolean))) '⊥)
  ; λ : check arg and result type
  (check-equal? (type '(λ () x) '((x Boolean))) '(→ Boolean))
  (check-equal? (type '(λ ((a : Number)) a)) '(Number → Number))
  (check-equal? (type '(λ ((a : Number) (b : Boolean)) a)) '(Number Boolean → Number))
  (check-equal? (type '(λ ((a : Number) (b : Boolean)) b)) '(Number Boolean → Boolean))
  ; λ : with multiple body expressions
  (check-equal? (type '(λ ((a : Number) (b : Boolean)) 1 "2" b)) '(Number Boolean → Boolean))   ; no ⊥ in body types
  (check-equal? (type '(λ ((a : Number) (b : Boolean)) name-in-env a) '((name-in-env ⊥))) '⊥)   ; has ⊥ in body types
  ; let : type check identifier with expression
  (check-equal? (type '(let ([a 1] [b #t]) a)) 'Number)
  (check-equal? (type '(let ([a 1] [b (λ ((a : Number)) a)]) b)) '(Number → Number))

  ; rec
  (check-equal? (type '(rec (f (arg1 : Number) (arg2  : Boolean) : Boolean) 1 2 arg2)) '(Number Boolean → Boolean))
  (check-equal? (type '(rec (f (arg1 : Number) : Number) 1)) '(Number → Number))
  (check-equal? (type '(rec (f (arg1 : Number) : Boolean) f)) '(Number → Boolean))   ; f saved to local-env

  ; if
  (check-equal? (type '(if #t 1 2)) 'Number)
  ; condition-expression not Boolean
  (check-equal? (type '(if 1 2 3)) '⊥)
  ; consequent and alternative not the same type
  (check-equal? (type '(if #t 1 "string")) '⊥)

  ; set!
  (check-equal? (type '(set! x 1) '((x Number))) 'Void)
  ; most local identifier with name not in env, or of different type than expression
  (check-equal? (type '(set! x 1) '((y Number))) '⊥)
  (check-equal? (type '(set! x 1) '((x Boolean))) '⊥)
  
  ; (f args)
  (check-equal? (type '((λ ((a : Number)) a) 1)) 'Number)
  ; (f arg1 args ...)
  (check-equal? (type '((λ ((a : Number) (b : Boolean)) b) 1 #t)) 'Boolean)
  (check-equal? (type '((λ ((b : Boolean) (a : Number)) a) #t 1)) 'Number)
  ; f is not function type
  (check-equal? (type '(1 1)) '⊥)
  ; f is a function type, but types of argument does not match function type's arugment types
  (check-equal? (type '((λ ((a : Number)) a) #f)) '⊥)

  )

; type : Expression → Type
; You may choose whatever representation for the environment that you like.
(define (type expr [env '()])
  (match expr
    [`(λ ((,<identifier> : ,<Type>) ...)
        ,<body-expression>
        ...
        ,<result-expression>)
     (define newly-bound-env (zip <identifier> <Type>))
     (define new-env (append newly-bound-env env))
     (define body-types (map (λ (x) (type x new-env)) <body-expression>)) ; a list 
     (define result-type (type <result-expression> new-env))              ; not a list
     (cond [(not (in? '⊥ body-types)) (append <Type> `(→ ,result-type))]
           [else '⊥])
     ]
    [`(let ([,<identifier> ,<expression>] ...)
        ,<body-expression>
        ...
        ,<result-expression>)
     (define expression-types (map (λ (x) (type x env)) <expression>))
     (define identifiers-and-types (zip2 <identifier> expression-types ':))
     (define function-evaluation-expression
       (list* (append `(λ ,identifiers-and-types) <body-expression> `(,<result-expression>))
              <expression>))
     (type function-evaluation-expression env)
     ]
    [`(rec (,<function-identifier> (,<identifier> : ,<Type>) ... : ,<result-Type>)
        ,<body-expression>
        ...
        ,<result-expression>)
     (define newly-bound-env `((,<function-identifier> ,<result-Type>)))
     (define new-env (append newly-bound-env env))
     (define identifiers-and-types (zip2 <identifier> <Type> ':))
     (define lambda-expression
       (append `(λ ,identifiers-and-types) <body-expression> `(,<result-expression>)))
     (type lambda-expression new-env)
     ]
    [`(if ,<condition-expression>
          ,<consequent-expression>
          ,<alternative-expression>)
     (define condition-type (type <condition-expression> env))
     (define consequent-type (type <consequent-expression> env))
     (define alternative-type (type <alternative-expression> env))
     (cond [(and (equal? 'Boolean condition-type) (equal? consequent-type alternative-type)) consequent-type]
           [else '⊥])
     ]
    [`(set! ,<identifier> ,<expression>)
     (define expression-type (type <expression> env))
     (define most-local-identifier-type (find-type-by-id env <identifier>))
     (cond [(equal? expression-type '⊥) '⊥]
           [(equal? expression-type most-local-identifier-type) 'Void]
           [else '⊥])
     ]
    [`(,<function-expression> ,<argument-expression> ...)
     (define function-type (type <function-expression> env))
     (define argument-types (map (λ (x) (type x env)) <argument-expression>))
     (match function-type
       [`(,<func-arg-Type> ... → ,<func-result-Type>)
        (cond [(equal? <func-arg-Type> argument-types) <func-result-Type>]
              [else '⊥])]
       [_ '⊥])
     ]
    [_ (cond [(symbol? expr) (find-type-by-id env expr)]
             [(boolean? expr) 'Boolean]
             [(number? expr)  'Number]
             [(string? expr)  'String]
             [else '⊥])]))
