#lang racket


#| A slight variant of the Memory Model tracing library from CSC324 2017 Fall: |#
(require "mm.jan-09.rkt")
(wait! #false) ; Uncomment to make it not wait for enter/return between steps.
(scale! 12) ; Font size.
(interleave! #false) ; Diagram it in a form closer to how we'll compile it.

#| Now running racket code in this file traces the evaluation while running it.
 It traces a subset of possible racket forms, but a superset of the core lambda calculus. |#


#| The next part adds details to the trace that we didn't track in CSC324.

 With these, each snapshot of the trace contains all the information necesary to continue
  evaluation. These snapshots reflect the information in memory that the compiled code
  will manage in order to execute.

 The implementation details are not important at the moment, except that it uses forms
  invisible to the tracer: this support code is not itself traced.

 This will eventually be merged into the CSC324 library with flags for whether to show it.

 For now, to get the extra tracing an expression needs to be inside a local that overrides
  function call to get the extra tracing. The examples later in the file illustrate this. |#
(module support racket
  (provide push! pop! show show-stack)
  (define result-stack (box '()))
  (define (push! v) (set-box! result-stack (list* v (unbox result-stack))))
  (define (pop!) (begin0 (first (unbox result-stack))
                         (set-box! result-stack (rest (unbox result-stack)))))
  (define (show . args) (displayln (apply ~a args)))
  (define (show-stack) (show "Result stack: " (unbox result-stack) ".")))
(require 'support)
(define-syntax-rule (app e1 e2)
  (let ([result e1])
    (show "• result of evaluating " (~s 'e1) ": " result)
    (show "• push result")
    (push! result)
    (show-stack)
    (let ([result e2])
      (show "• result of evaluating " (~s 'e2) ": " result)
      (let ([f (pop!)])
        (show "• pop: " f)
        (show-stack)
        (show "• call: " f)
        (f result)))))


#| Recall the core Lambda Calculus:
     id
     (λ (id) e)
     (e1 e2) |#

#| Our evaluation model, that easily maps to a machine computation.

 Closure:
     Consists of:
       a record storing a function
       an environment
     Implementation:
       Data structure containing a pointer to function, and representation of env
       Referencing env binds non-local names to corresponding variables in the lexical env
       at the time closure is created, additionally extending their lifetime to at least as
       long as the lifetime of the closure itself
     Memory:
       Closure requires free variables it references survive the enclosing function's execution
       So those variables must be allocated so that they persis until no longer needed, via heap allocation
       In C++11, closure function yield undefined behavior for accessing freed automatic variables
       It is difficult to implement fnctions as first class objects in stack-based programming language like C/C++
 Environment:
     a mapping associating each free variable of function, used locally but defined in enclosing
     scope, with value or reference to which the name was bound when the closure was created
 

 What we keep track of:

   Growing tree of environments.
     Each environment, except the root, contains a parameter and its value, which is a closure.
     Initially a single empty root node, which we'll refer to as •.
   Current environment: one of the environments in that tree.
     Initially the root environment.
   Call stack of environments: a stack of environments from that tree.
     Initially empty.

   Current result, which is a closure.
     Initially meaningless.
   Stack of results, which is a stack of closures.
     Initially empty.

   Growing set of closures.
     Each closure is a pair of a λ expression and the environment that was current when it was made.
     Initially empty.

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
     Evaluate e2. (note current result now is value of e2)
     Pop to get the closure to call, let's refer to it as λf.
     Add a new environment E<n> to the tree of environments, under λf's environment, with the id
      from λf's λ expression and the current result [which is the value of e2].
     Push the current environment onto the call stack.
     Set the current environment to E<n>.
     Evaluate the body of λf's λ expression.
     Pop the call stack into the current environment. |#

#;x ; Invalid.

#;(λ (x) x)
; Add a closure λ0 to the set of closures.
;   λ0 is (λ (x) x) paired with the root environment •.
; Set the current result to λ0.

#;(x x)
; Evaluate x. Invalid.

#;((λ (x) x) x)
; Evaluate (λ (x) x).
;   ⋯
; Evaluate x. Invalid.

#;((λ (x) x) (λ (y) y))
; Evaluate (λ (x) x).
;   ⋯ see earlier example ⋯
; Push current result, λ0, onto the result stack.
; Evaluate (λ (y) y).
;   ⋯ similar ⋯
; Set of closures is {λ0 = [(λ (x) x) •], λ1 = [(λ (y) y) •]}.
; Current result is λ1.
; Pop to get the closure to call, it's λ0.
; Add a new environment E1 to the tree of environments, under λ0's environment •, with x and λ1. (i.e. x is bound to λ1)
; Push the current environment • onto the call stack.
; Set the current environment to the new environment E1.
; Evaluate x.
; Pop the call stack into the current environment.

; With the extra tracing:
#;(local [(define-syntax-rule (#%app e1 e2) (app e1 e2))]
    ((λ (x) x) (λ (y) y)))
; Notice the double-boxing around an environment: it represents the current environment.

#| Another example. |#

#;(local [(define-syntax-rule (#%app e1 e2) (app e1 e2))]
    ((λ (x) (λ (y) (x y))) (λ (z) z)))

#| Another example. |#

(local [(define-syntax-rule (#%app e1 e2) (app e1 e2))]
    (((λ (x) (λ (y) (x y))) (λ (z) z))
     (λ (a) a)))

