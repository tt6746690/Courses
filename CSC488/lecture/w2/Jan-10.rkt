#lang racket


#| A slight variant of the Memory Model tracing library from CSC324 2017 Fall: |#
(require "mm.jan-10.rkt")
#;(wait! #false) ; Uncomment to make it not wait for enter/return between steps.
(scale! 12) ; Font size.
(interleave! #false) ; Diagram it in a form closer to how we'll compile it.

#| Now running racket code in this file traces the evaluation while running it.

 It traces a subset of possible racket forms, but a superset of the core lambda calculus.

 Details of function call are not shown, unless we import a new function call: |#

(require (only-in "mm.jan-10.rkt" [app #%app])) ; Don't worry about how this overrides function call.

#| That overrides unary function call, so that the tracing of core Lambda Calculus expressions
    contains all the information necessary to continue evaluation.

 The diagrams already show all closures and environments, and double-boxes the current environment.
 The textual output shows the the call stack and result stack, and mentions the use of the temporary
  “result” location. |#


#| Recall the core Lambda Calculus:
     id
     (λ (id) e)
     (e1 e2)

 That's a definition by structural induction, although it leaves out some details.

 If you aren't familiar with the Lambda Calculus, here's a complete description of expressions/terms.
 A Lambda Calculus expression is one of:
   id, where id is any reasonable “name” [called an “identifier” in Programming Language Theory].
   (λ (id) e), where id is an identifier and e is an expression.
   (e1 e2), where e1 and e2 are expressions. |#

; Systematically generate some expressions, by looping over the rules, trying to generate a
;  “minimally new” expression each time:
#;x
#;(λ (x) x)
#;(x x)
#;y
#;(λ (x) y)
#;(x y)
#;z
#;(λ (x) (x x))
#;((λ (x) x) x)
#;⋯

; Make sure you can systematically generate examples for *any* definition by structural induction.

#| Our evaluation model, that easily maps to a machine computation.

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
     Evaluate e2.
     Pop to get the closure to call, let's refer to it as λf.
     Add a new environment E<n> to the tree of environments, under λf's environment, with the id
      from λf's λ expression and the current result [which is the value of e2].
     Push the current environment onto the call stack.
     Set the current environment to E<n>.
     Evaluate the body of λf's λ expression.
     Pop the call stack into the current environment. |#

#;x ; Invalid.

(λ (x) x)
; Add a closure λ0 to the set of closures.
;   λ0 is (λ (x) x) paired with the root environment •.
; Set the current result to λ0.

; Notice the double-boxing around an environment: it represents the current environment.

#;(x x)
; Evaluate x. Invalid.

#;((λ (x) x) x)
; Evaluate (λ (x) x).
;   ⋯
; Evaluate x. Invalid.

#;((λ (x) x) (λ (y) y)) ; Uncomment this and run.
; Evaluate (λ (x) x).
;   ⋯ see earlier example ⋯
; Push current result, λ0, onto the result stack.
; Evaluate (λ (y) y).
;   ⋯ similar ⋯
; Set of closures is {λ0 = [(λ (x) x) •], λ1 = [(λ (y) y) •]}.
; Current result is λ1.
; Pop to get the closure to call, it's λ0.
; Add a new environment E1 to the tree of environments, under λ0's environment •, with x and λ1.
; Push the current environment • onto the call stack.
; Set the current environment to the new environment E1.
; Evaluate x.
; Pop the call stack into the current environment.

#| Another example: uncomment it and run. |#
#;((λ (x) (λ (y) (x y))) (λ (z) z))

#| Another example: uncomment it and run. |#
#;(((λ (x) (λ (y) (x y))) (λ (z) z))
   (λ (a) a))







#; ((λ (x) x) 488)
; Compilation
#; ((λ0 (x) x)
    (variable 0)) ; the immediate parameter
#; {(closure λ0)
    (push-result)
    (result 488)
    (call)}

; Note envs are separete from lambdas in the diagram.

(((λ (x) (λ (y) x)) 488) 2107)
; Compilation
#; (λ0 (closure λ1))
#; (λ1 (variable 1))  ; look up 1 pointer to prev env to find variable
#; ((closure λ0)
    (push-result)
    (result 488)
    (call)
    (push-result)
    (result 2107)
    (call))

; make a variable, and do a computation on the variable in its scope
#;(let (c 48)
    (λ (x) c))
; body of computation parameterized on value of c

#; ((λ (c)
      (λ (x) c))
    488)

'((λ (c)
    (λ (x) c))
  488)

