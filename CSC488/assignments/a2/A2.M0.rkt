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

; L0 languages
#;(L0: λ (<id>) <e>)
#;(L0: app <e1> <e2>)
#;(L0: var <id>)
#;(L0: datum <i>)
#;(L0: set! <id> <e>)
#;(L0: if <e1> <e2> <e3>)

(define-syntax-rule
  (define-transformer id name 
    clause  
    ...)
  (define id (transformer 'name (λ (e) (match e
                                         clause
                                         ...)))))

(define (T-f transformer-name e)
  ((transformer-function transformer-name) e))

(module+ test (require rackunit)
  
  ; Test transformer for *id* *datum* set! if
  ;   (*id* *<id>*) → (L0: var *<id>*)
  ;   (*datum* *<i>*) → (L0: datum *<i>*)
  ;   (set! *<id>* *<e>*) → (L0: set! *<id>* *<e>*)
  ;   (if! *<e1>* *<e2>* *<e3>*) → (L0: if *<e1>* *<e2>* *<e3>*)
  (check-equal? (T-f T:*id* '(*id* a))
                '(L0: var a))
  (check-equal? (T-f T:*datum* '(*datum* 10))
                '(L0: datum 10))
  (check-equal? (T-f T:set! '(set! x 1))
                '(L0: set! x 1))
  (check-equal? (T-f T:if '(if e1 e2 e3))
                '(L0: if e1 e2 e3))
  
  )

; *id* *datum* set! if
; --------------------
; The following two special cases should expand to integers 0 and 1:
#;(*id* false)
#;(*id* true)
; Otherwise, they are straightforward untagged versions of L0 expressions.
; Transform those directly to their L0 form.
(define-transformer T:*id* *id*
  [`(,_ false) '(L0: datum 0)]
  [`(,_ true) '(L0: datum 1)]
  [`(,_ ,<id>) `(L0: var ,<id>)])
(define-transformer T:*datum* *datum*
  [`(,_ ,<i>) `(L0: datum ,<i>)])
(define-transformer T:set! set!
  [`(,_ ,<id> ,<e>) `(L0: set! ,<id> ,<e>)])
(define-transformer T:if if
  [`(,_ ,<e1> ,<e2> ,<e3>) `(L0: if ,<e1> ,<e2> ,<e3>)])

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

(define (repeat f v n)
  (cond [(equal? n 0) v]
        [else (repeat f (f v) (- n 1))]))

(module+ test
  (define T-n (curry T-f T:λ))
  #;(check-equal? (T-f T:λ '(λ (x) (var x)))
                  '(L0 λ (var x)))
  ; allow zero parameters
  #;(check-equal? (T-f T:λ '(λ () (datum 1)))
                  '(L0: λ (_) (block (datum 1))))
  ; allow more than one body exprs
  (check-equal? (repeat (curry T-f T:λ) '(λ () (datum 1) (datum 2)) 1)
                '(λ () (block (datum 1) (datum 2))))
  (check-equal? (repeat (curry T-f T:λ) '(λ () (datum 1) (datum 2)) 2)
                '(λ (_) (block (datum 1) (datum 2))))
  (check-equal? (repeat (curry T-f T:λ) '(λ () (datum 1) (datum 2)) 3)
                '(L0: λ (_) (block (datum 1) (datum 2))))
  ; 1 arg
  (check-equal? (repeat (curry T-f T:λ) '(λ (x) (datum 1)) 1)
                '(λ (x) (block (datum 1))))
  (check-equal? (repeat (curry T-f T:λ) '(λ (x) (datum 1)) 2)
                '(L0: λ (x) (block (datum 1))))
  ; 2 arg 
  (check-equal? (repeat (curry T-f T:λ) '(λ (x y) (+ x y)) 1)
                '(λ (x y) (block (+ x y))))
  (check-equal? (repeat (curry T-f T:λ) '(λ (x y) (+ x y)) 2)
                '(λ (x) (λ (y) (block (+ x y)))))
  (check-equal? (repeat (curry T-f T:λ) '(λ (x y) (+ x y)) 3)
                '(λ (x) (block (λ (y) (block (+ x y))))))
  (check-equal? (repeat (curry T-f T:λ) '(λ (x y) (+ x y)) 4)
                '(L0: λ (x) (block (λ (y) (block (+ x y))))))
  ; body not in bracket
  (check-equal?  (repeat (curry T-f T:λ) '(λ (x y) x y (+ x y)) 1)
                 '(λ (x y) (block x y (+ x y))))
  )


(define-transformer T:λ λ
  ; wrap body in 'block', if
  ;   • first body entry not a list,
  ;   • if first body entry is a list, not tagged by block already
  [`(λ ,<id> ,<body> ...) #:when (not (and (list? (first <body>))
                                           (equal? (first (first <body>))
                                                   'block)))
                          `(λ ,<id> (block . ,<body>))]
  ; unary single-body-expr → L0 form
  [`(λ (,<id>) ,<body> ...) `(L0: λ (,<id>) . ,<body>)]
  ; adding dummy if 0 parameter
  [`(λ () ,<body> ...) `(λ (_) . ,<body>)]
  ; curry for two or more parameters
  [`(λ (,<id> ... ,<id-last>) ,<body> ...)
   `(λ ,<id> (λ (,<id-last>) . ,<body>))])


; *app*
; -----
; Extends L0's app by allowing zero arguments, or more than one argument.
; Transform the form with more than one argument into its curried equivalent.
; Transform the no-argument form into a one-argument form with a dummy argument [see ‘block’].
; Transform the unary form to the L0 form.

(module+ test
  ; more than 1 arg
  (check-equal? (repeat (curry T-f T:*app*) '(*app* f a b) 1)
                '(*app* (*app* f a) b))
  (check-equal? (repeat (curry T-f T:*app*) '(*app* f a b) 2)
                '(L0: app (*app* f a) b))
  ; zero arg
  (check-equal? (repeat (curry T-f T:*app*) '(*app* e) 1)
                '(*app* e (block)))
  (check-equal? (repeat (curry T-f T:*app*) '(*app* e) 2)
                '(L0: app e (block)))
  ; unary → L0 form
  (check-equal? (repeat (curry T-f T:*app*) '(*app* f e) 1)
                '(L0: app f e))
  )

#; (f e1 e2) ; ((f e1) e2)
#; (e)       ; (e _)
#; (f e)     ; (L0: app f e)

(define-transformer T:*app* *app*
  [`(,_ ,<f>) `(*app* ,<f> (block))]
  [`(,_ ,<f> ,<e>) `(L0: app ,<f> ,<e>)]
  [`(,_ ,<f> ,<e0> ,<e1> ...)
   `(*app* (*app* ,<f> ,<e0>) . ,<e1>)])

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

(module+ test
  ; 0 expression
  (check-equal? (repeat (curry T-f T:block) '(block) 1)
                '(L0: datum 0))
  ; 1 expression
  (check-equal? (repeat (curry T-f T:block) '(block e0) 1)
                'e0)
  ; 2 expressions
  (check-equal? (repeat (curry T-f T:block) '(block e0 e1) 1)
                '(let ([_ e0]) (block e1)))
  ; 3 expressions
  (check-equal? (repeat (curry T-f T:block) '(block e0 e1 e2) 1)
                '(let ([_ e0]) (block e1 e2)))
  )

(define-transformer T:block block
  [`(,_) '(L0: datum 0)]
  [`(,_ ,<e>) <e>]
  [`(,_ ,<e0> ,<e1> ...)
   `(let ([_ ,<e0>])
      (block . ,<e1>))])


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

(module+ test
  ; 1 (id, init) pair
  (check-equal? (repeat (curry T-f T:let) '(let ([x 1]) x) 1)
                '((λ (x) x) 1))
  (check-equal? (expand '(let ([x 1]) x) Ts)
                '(L0: app (L0: λ (x) (L0: var x)) (L0: datum 1)))
  ; >1 (id, init) pair
  #; (let ([x 1] [y 2]))
  #; ((λ (x y) x y) 1 2)
  #; (((λ (x)
         (λ (y)
           ((λ (_) y)
            x)))
       1)
      2)
  (check-equal? (repeat (curry T-f T:let) '(let ([x 1] [y 2]) x y) 1)
                '((λ (x y) x y) 1 2))
  (check-equal? (expand '(let ([x 1] [y 2]) x y) Ts)
                '(L0: app
                      (L0: app
                           (L0: λ (x)
                                (L0: λ (y)
                                     (L0: app (L0: λ (_) (L0: var y)) (L0: var x))))
                           (L0: datum 1))
                      (L0: datum 2)))
  #; (((λ (x)
         (λ (y) x))
       1)
      2)
  )



(define-transformer T:let let
  [`(,_ ([,<id> ,<init>] ..1)
        ,<body> ..1)
   `((λ ,<id> . ,<body>) . ,<init>)])



; local
; -----
#;(local [(define (<f-id> <id> ...)
            <f-body>
            ...+)
          ...+]
    <body>
    ...+)
; convert to
#; (let ([<f-id> (block)] ...+)
     (block (set! <f-id>
                  (λ (<id> ...) <f-body> ...+))
            ...+
            body ...+)) 
; Introduces the distinctly named local <f-id>s into scope, to functions created in that scope,
;  then evaluates the <body>s as a block.
;
; Transform using the standard LC+set! transformation: to an expression that initializes
;  all the <f-id>s to dummy values, sets them to their functions, then evaluates the body.

(module+ test
  (check-equal? (repeat (curry T-f T:local)
                        '(local [(define (f0 arg0 arg1)
                                   arg1 arg0)
                                 (define (f1 arg2 arg3)
                                   arg3 arg2)]
                           (f0 1))
                        1)
                '(let ([f0 (block)] [f1 (block)])
                   (block
                    (set! f0 (λ (arg0 arg1) arg1 arg0))
                    (set! f1 (λ (arg2 arg3) arg3 arg2))
                    (f0 1))))
  )

(define-transformer T:local local
  [`(local [(define (,<f-id> ,<id> ...)
              ,<f-body>
              ..1)
            ..1]
      ,<body>
      ..1)
   `(let ,(map (λ (f-id) `(,f-id (block))) <f-id>)
      ,(append '(block)
               (map (λ (f-id id f-body)
                      `(set! ,f-id (λ ,id . ,f-body)))
                    <f-id> <id> <f-body>)
               <body>))])

; and or
; ------
#;(and <e0> <e> ...+)
#;(or  <e0> <e> ...+)
; Standard short-circuiting operators for two or more boolean expressions.
; Transform to ‘if’s or ‘cond’s.
#;(if <e0> (and <e> ...) false)        ; (and e1 e ...)    
#;(if <e1> (if <e2> true false) false) ; (and e1 e2)
#;(if <e0> true (or <e> ...))          ; (or e1 e ...) 
#;(if <e1> true (if <e2> true false))  ; (or e1 e2)

(module+ test
  ; and
  (check-equal? (repeat (curry T-f T:and) '(and e0 e1) 1)
                '(if e0 (if e1 true false) false))
  (check-equal? (repeat (curry T-f T:and) '(and e0 e1 e2) 1)
                '(if e0 (and e1 e2) false))
  (check-equal? (expand '(and e0 e1 e2) Ts)
                '(L0: if (L0: var e0)
                      (L0: if
                           (L0: var e1)
                           (L0: if (L0: var e2)
                                (L0: datum 1)
                                (L0: datum 0))
                           (L0: datum 0))
                      (L0: datum 0)))
  ; or
  (check-equal? (repeat (curry T-f T:or) '(or e0 e1) 1)
                '(if e0 true (if e1 true false)))
  (check-equal? (repeat (curry T-f T:or) '(or e0 e1 e2) 1)
                '(if e0 true (or e1 e2)))
  (check-equal? (expand '(or e0 e1 e2) Ts)
                '(L0: if (L0: var e0)
                      (L0: datum 1)
                      (L0: if (L0: var e1)
                           (L0: datum 1)
                           (L0: if (L0: var e2)
                                (L0: datum 1)
                                (L0: datum 0)))))
  )


(define-transformer T:and and
  [`(,_ ,<e0> ,<e1>)
   `(if ,<e0> (if ,<e1> true false) false)]
  [`(,_ ,<e0> ,<e> ...)
   `(if ,<e0> (and . ,<e>) false)])

(define-transformer T:or or
  [`(,_ ,<e0> ,<e1>)
   `(if ,<e0> true (if ,<e1> true false))]
  [`(,_ ,<e0> ,<e> ...)
   `(if ,<e0> true (or . ,<e>))])


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

(module+ test
  (check-equal? (repeat (curry T-f T:cond) '(cond [cond1 1 2]) 1)
                '(when cond1 1 2))
  (check-equal? (repeat (curry T-f T:cond) '(cond [cond1 1 2] [cond2 4 5]) 1)
                '(if cond1 (block 1 2) (cond (cond2 4 5))))
  (check-equal? (repeat (curry T-f T:cond) '(cond [else 3 4]) 1)
                '(block 3 4))
  )


(define-transformer T:cond cond
  [`(,_ [else ,<else-result>
              ..1])
   `(block . ,<else-result>)]
  [`(,_ [,<condition> ,<result>
                      ..1])
   `(when ,<condition> . ,<result>)]
  [`(,_ [,<condition> ,<result>
                      ..1]
        ..1)
   `(if ,(first <condition>)
        (block . ,(first <result>))
        (cond . ,(map (λ (cond result) `[,cond . ,result])
                      (rest <condition>)
                      (rest <result>))))])


; when
; ----
#;(when <condition>
    <body>
    ...+)
; If boolean <condition> is true evaluates the <body>s as a block, otherwise produces a dummy value.

(module+ test
  (check-equal? (repeat (curry T-f T:when) '(when e0 b0 b1) 1)
                '(if e0 (block b0 b1) (block)))
  )

(define-transformer T:when when
  [`(,_ ,<condition>
        ,<body>
        ..1)
   `(if ,<condition>
        (block . ,<body>)
        (block))])


; while
; -----
#;(while <condition>
         <body>
         ...+)
; A standard while loop.
; Transform to a recursive no-argument function that is immediately called.


(module+ test
  (check-equal? (repeat (curry T-f T:while) '(while condition b0 b1) 1)
                '(local
                   ((define (*<loop>*)
                      (when condition b0 b1 *<loop>*)))
                   (*<loop>*)))
  )

(define-transformer T:while while
  [`(,_ ,<condition>
        ,<body>
        ..1)
   `(local [(define (*<loop>*)
              ,(append '(when)
                       `(,<condition>)
                       <body>
                       '(*<loop>*)))]
      (*<loop>*))])



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

(module+ test
  ; returnable
  (check-equal? (repeat (curry T-f T:returnable) '(returnable e0 e1) 1)
                '(call/ec (λ (return) e0 e1)))
  (check-equal? (expand '(returnable e0 e1) Ts)
                '(L0: app (L0: var call/ec)
                      (L0: λ (return)
                           (L0: app (L0: λ (_) (L0: var e1))
                                (L0: var e0)))))
  ; breakable
  (check-equal? (repeat (curry T-f T:breakable) '(breakable (while condition e0 e1)) 1)
                '(call/ec (λ (break) (while condition e0 e1))))
  ; (expand '(breakable (while condition body)) Ts)
  ; continuable
  (check-equal? (repeat (curry T-f T:returnable) '(returnable e1 e2) 1)
                '(call/ec (λ (return) e1 e2)))
  )


; (λ () (returnable <e> ...))
; (λ () (call/ec (λ (return) <e> ...))

; (breakable (while condition <e> ...) ...)
; (call/ec (λ (break) (while condition <e> ...) ...) 

; (while condition (continuable <e> ...) ...)
; (while condition (call/ec (λ (continue) <e> ...)) ...)

(define-transformer T:returnable returnable
  [`(,_ ,<e> ..1)
   `(call/ec (λ (return) . ,<e>))])
(define-transformer T:breakable breakable
  [`(,_ ,<e> ..1)
   `(call/ec (λ (break) . ,<e>))])
(define-transformer T:continuable continuable
  [`(,_ ,<e> ..1)
   `(call/ec (λ (continue) . ,<e>))])


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
           (define (not b) (if b false true))
           
           ; Arithmetic  + < * available
           ; ----------
           ; (- a b) : the difference between a and b
           (define (- a b) (+ a (⊖ b)))
           ; (⊖ a) : the negative of a
           (define (⊖ a) (* -1 a))
           ; (> a b) : whether a is greater than b
           (define (> a b) (< b a))
           ; (>= a b) : whether a is greater than or equal to b
           (define (>= a b) (not (< a b)))
           ; (= a b) : whether a is equal to b
           (define (= a b) (and (>= a b) (not (> a b))))
           ]
     ,e))



