#lang racket #| Compile L1 to s-expression-machine language L2 |#

(provide (struct-out compiled:L2) L1→L2)
(module+ test (require rackunit))

#| Language L2

 This is a machine-like language, for a machine that supports enough to track the extended information
  in the memory model. Along with the support to model the LC, it has support for variable mutation,
  integer constants, and a few named lambdas.

 There are six types of statement in L2.
 All expression results move through a special result location. |#

#;(L2: closure <name>) ; λ
; Create a unary closure from the compiled body named <name>.
; Leave a reference to it in result.

#;(L2: variable <n>) ; var
; Look up the variable <n> environments up from the current environment, put its value in result.

; Put the value of result on the stack.
#;(L2: push_result) ; app

; Call the closure that is on the stack, with the argument that is in result.
#;(L2: call) ; app

; Set the variable <n> environments up from the current environment, to the value of result.
#;(L2: set <n>) ; set!

; Put integer <i> in result.
#;(L2: set_result <i>) ; datum
 

#| Compiling L1 to L2

 From your tracing examples, you understand how to track the memory model via the above operations.

 Compiling an expression in L1 produces:
  • code: a list of L2 statements for the expression
  • λs: a list of two-element lists (<name> <code>) with the <name>s of the λs in the expression
     paired with the code for their bodies

 The following summarizes your understanding.
 These *descriptions* are slightly informal, in particular they blur the distinction between
  lists and individual elements of a list. Don't blindly turn them into an implementation! |#


#;{(L1: λ <n> <e>) →
                   code: (L2: closure lambda_<n>)  ; L2 statements
                   λs: {(lambda_<n> <code-for-e>)  ; L2 statements
                        <λs-for-e>}}

#;{(L1: app <e1> <e2>) →
                       code: {<code-for-e1>
                              (L2: push_result)
                              <code-for-e2>
                              (L2: call)}
                       λs: {<λs-for-e1>
                            <λs-for-e2>}}

#;{(L1: set! <n> <e>) →
                      code: {<code-for-e>
                             (L2: set <n>)}
                      λs: {<λs-for-e>}}

; Each of the following produce a single L2 statement and no λs:
#;{(L1: var <n>) → (L2: variable <n>)}
#;{(L1: var +) → (L2: closure make_add)}
#;{(L1: var *) → (L2: closure make_multiply)}
#;{(L1: var <) → (L2: closure make_less_than)}
#;{(L1: datum <i>) → (L2: set_result <i>)}


#| L1→L2

 Each expression produces an instance of struct compiled:L2, which has two fields for code and λs. |#

(struct compiled:L2 (code λs) #:transparent)

(module+ test
  ; Along with making a constructor named ‘compiled:L2’ and also a pattern with that name,
  ;  the struct declaration implicitly defines two field accessor functions: ‘compiled:L2-code’
  ;  and ‘compiled:L2-λs’, which you might prefer sometimes instead of needing to name results
  ;  via pattern matching.
  
  ;• code: a list of L2 statements for the expression
  ;• λs: a list of two-element lists (<name> <code>) with the <name>s of the λs in the expression
  ;  paired with the code for their bodies
  (define c (compiled:L2 '((L2: set_result 123)) '((lambda_0 ((L2: variable 0))))))
  (check-equal? (match c [(compiled:L2 some-code some-λs) some-code])
                (compiled:L2-code c))
  (check-equal? (match c [(compiled:L2 some-code some-λs) some-λs])
                (compiled:L2-λs c)))

; Produce a symbol of the form lambda_<n>.
(define (lambda_ n)
  (local-require (only-in racket/syntax format-symbol))
  (format-symbol "lambda_~a" n))

(module+ test
  (check-equal? (lambda_ 123) 'lambda_123))

#|
 Evaluation:
   id
     Valid only during the evaluation of a λ body, during a call to the λ, where id is one of
      the parameters in the chain of environments from the closure's environment upwards.
     Set the current result to be the value of id in that environment.
   (λ (id) e)
     Add a closure λ<n> pairing this λ expression and current environment, to the set of closures.
     Set the current result to be λ<n>.
   (e1 e2)
     Evaluate e1.
     Push the current result [the value of e1] onto the stack of results.
     Evaluate e2.
     Pop to get the closure to call, let's refer to it as λf.
     Add a new environment E<n> to the tree of environments, under λf's environment, with the id
      from λf's λ expression and the current result [which is the value of e2].
     Push the current environment onto the call stack.
     Set the current environment to E<n>.
     Evaluate the body of λf's λ expression.
     Pop the call stack into the current environment. |#

; L1 Language
#;(L1: λ <n> <e>)     ; The nth lambda in an L1 expression.
#;(L1: app <e1> <e2>) ; Same meaning as in L0.
#;(L1: var <n>)       ; Reference to a variable <n> scopes up.
#;(L1: var <id>)      ; Free/unbound/open variable reference to <id>.
#;(L1: set! <n> <e>)  ; Set the variable <n> scopes up to the value of <e>.
#;(L1: datum <i>)     ; Same meaning as in L0.


(module+ test
  ; test L1 producing no lambdas
  (check-equal? (L1→L2 '(L1: var 0))
                (compiled:L2 '((L2: variable 0)) '()))
  (check-equal? (L1→L2 '(L1: datum 1))
                (compiled:L2 '((L2: set_result 1)) '()))
  (check-equal? (L1→L2 '(L1: var +))
                (compiled:L2 '((L2: closure make_add)) '()))
  (check-equal? (L1→L2 '(L1: var *))
                (compiled:L2 '((L2: closure make_multiply)) '()))
  (check-equal? (L1→L2 '(L1: var <))
                (compiled:L2 '((L2: closure make_less_than)) '()))
  ; test λ
  (check-equal? (L1→L2 '(L1: λ 0 (L1: var 0)))
                (compiled:L2 '((L2: closure lambda_0))
                             '((lambda_0 ((L2: variable 0))))))
  (check-equal? (L1→L2 '(L1: λ 1 (L1: λ 0 (L1: var 1))))
                (compiled:L2 '((L2: closure lambda_1))
                             '((lambda_1 ((L2: closure lambda_0)))
                               (lambda_0 ((L2: variable 1))))))

  ; test set!
  (check-equal? (L1→L2 '(L1: set! 3 (L1: var 2)))
                (compiled:L2 '((L2: variable 2)
                               (L2: set 3))
                             '()))
  ; set! with λ
  (check-equal? (L1→L2 '(L1: λ 1 (L1: set! 0 (L1: λ 0 (L1: var 0)))))
                (compiled:L2 '((L2: closure lambda_1))
                             '((lambda_1 ((L2: closure lambda_0)
                                          (L2: set 0)))
                               (lambda_0 ((L2: variable 0))))))
  ; test app
  (check-equal? (L1→L2 '(L1: app (L1: λ 0 (L1: var 0)) (L1: datum 2)))
                (compiled:L2 '((L2: closure lambda_0)
                               (L2: push_result)
                               (L2: set_result 2)
                               (L2: call))
                             '((lambda_0 ((L2: variable 0))))))
  )




(define (L1→L2 e)
  (match e
    ; λ
    [`(L1: λ ,<n> ,<e>)
     (define lambda-name (lambda_ <n>))
     (define compiled-<e> (L1→L2 <e>))
     (compiled:L2 `((L2: closure ,lambda-name))
                  (append `((,lambda-name ,(compiled:L2-code compiled-<e>)))
                          (compiled:L2-λs compiled-<e>)))]
    ; app
    [`(L1: app ,<e1> ,<e2>)
     (define compiled-<e1> (L1→L2 <e1>))
     (define compiled-<e2> (L1→L2 <e2>))
     (compiled:L2 (append (compiled:L2-code compiled-<e1>)
                          '((L2: push_result))
                          (compiled:L2-code compiled-<e2>)
                          '((L2: call)))
                  (append (compiled:L2-λs compiled-<e1>)
                          (compiled:L2-λs compiled-<e2>)))
     ]
    ; set!
    [`(L1: set! ,<n> ,<e>)
     (define compiled-<e> (L1→L2 <e>))
     (compiled:L2 (append (compiled:L2-code compiled-<e>)
                          `((L2: set ,<n>)))
                  (compiled:L2-λs compiled-<e>))]
    ; L1 statement that produce no lambdas
    [`(L1: var +) (compiled:L2 '((L2: closure make_add)) '())]
    [`(L1: var *) (compiled:L2 '((L2: closure make_multiply)) '())]
    [`(L1: var <) (compiled:L2 '((L2: closure make_less_than)) '())]
    [`(L1: var ,<n>)
     (compiled:L2 `((L2: variable ,<n>)) `())]
    [`(L1: datum ,<i>)
     (compiled:L2 `((L2: set_result ,<i>)) '())]
    [_ (compiled:L2 '() '())]))
