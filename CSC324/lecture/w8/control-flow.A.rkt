#lang racket #| Understanding Control Flow: Continuations. |#

#| Some “interesting” control flow:

 • in loops : break and continue
 • exceptions : throw / raise
 • generators : yield
 • logic search : backtracking and cut
 • web interaction : continuation capturing web servers

 Some of the languages with access to continuations at run time:

    Ruby
    Javascript Rhino
    Haskell
    C# / VB.Net

 Continuation-based web servers are available for many languages. |#

#| The “continuation” of an expression is:

     The control flow that would occur *after* the expression is evaluated, including
      what will be done with the *result value* of the expression. (this is important) 

 Note the “*after*” : the continuation does *not* include any control flow *inside* the expression.

 Let's review two common forms of “local” control flow. |#

(local []
  ; Sequential:
  (println 'a)
  ; ★ The "continuation" of (println 'a) is ★ onward: it does *not* include the printing of 'a.
  (println 'b)
  ; ★★ The continuation of (println 'b) is ★★ onward: it does *not* include the printing of 'b.
  )

(local []
  (define r0 (+ 1 20))
  (define r1 (+ 300 4000))
  (+ r0 r1))

; Control flow of strict/eager compositional code is not left-to-right top-to-bottom.
; There is the stacking up of the waiting calls, as part of a traversal of the expression “tree”.
(+ (+ 1 20)
   ; ★★★
   (+ 300 4000))

; The continuation of (+ 1 20) is ★★★, which includes going back to the waiting outer addition.
; The continuation is: add to the result of adding 300 and 4000.
; The continuation does not include the evaluation of (+ 1 20), and is thought of more precisely as
;  the part of control *after* (+ 1 20), that is *waiting* for that result, to then evaluate
;  (+ 300 400) and add the two results.
#;(λ (e) (<procedure:+> e
                        (+ 300 4000)))
   
; The outer identifier ‘+’ has been looked up, which “<procedure:+>” is meant to suggest.

; The continuation of (+ 300 4000) is something like:
#;(λ (e) (<procedure:+> 21
                        e))

; In general, the function expression is evaluated before the argument expressions.

((if (zero? (random 2)) + -) (random 324))

; Add side-effects to see the order:
((local []
   (displayln "Evaluating the function expression.")
   #;0 ; Uncomment to find out when the check that it's a function occurs.
   (if (zero? (random 2)) + -))
 (local []
   (displayln "Evaluating the argument expression.")
   (random 324)))

#;(f-expr a-expr ...)
;  1. Evaluate f-expr, a-expr, ..., in order, producing values.
;  2. Error if f-expr is not a function.
;  3. Otherwise, pass a-expr ... values to the value of f-expr.
