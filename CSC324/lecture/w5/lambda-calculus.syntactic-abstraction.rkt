#lang racket ; Choose "The Racket language" from the Language dialog in DrRacket.

#| Syntactic Abstraction.

 A fancy term for something you never noticed because it's been used for thousands of
  years, billions of times every day in the civilized world, without fuss or comment.

 Except most programmers forget to tell computers about it, so programmers have to
  write "boiler plate", follow "design patterns", etc, to generate a lot of code
  that a computer could trivially create for them. |#


#| Church restricted functions to be unary in the Lambda Calculus, because that makes
    it simpler to study and prove properties of, and because he could just say: |#

; When I talk about an expression and write
#;(λ (<id-0> <id-1> <id> ...) <expr>)
;  it is to save me *writing out* the slightly longer *actual expression*
#;(λ (<id-0>) (λ (<id-1> <id> ...) <expr>))
;  where of course if that inner λ expression uses this *shorthand* then it too stands for
;  an expanded *actual* expression.

; Note: here "..." is Kleene-*, so "<id> ..." is like "<id>*", i.e. zero or more <id>s.

; And Church said, when I write
#;(<f-expr> <expr-0> <expr-1> <expr> ...)
;  it is shorthand for me writing the *actual expression*:
#;((<f-expr> <expr-0>) <expr-1> <expr> ...)

; Let's tell Racket about this shorthand.
; The built-in λ already has a non-unary meaning, so we'll define a *shorthand* tagged
;  with the name "Lambda" to be *rewritten* at *compile time* to an *expression* written
;  with unary λ *expressions*.

(define-syntax Lambda
  (syntax-rules ()
    [(Lambda (<id-0> <id-1> <id> ...) <expr>)             ; match function with >=2 function
     (Lambda (<id-0>) (Lambda (<id-1> <id> ...) <expr>))] ; return a function with one less arg
    [(Lambda (<id>) <expr>)                               ; match unary function
     (λ (<id>) <expr>)]))

; For example, the following is not directly an expression in the Lambda Calculus
(Lambda (condition consequent alternative)
        ((condition consequent) alternative))
;  it was just to save the trouble of typing out
(Lambda (condition)
        (Lambda (consequent alternative)
                ((condition consequent) alternative)))
;  which saves the trouble of typing out
(Lambda (condition)
        (Lambda (consequent)
                (Lambda (alternative)
                        ((condition consequent) alternative))))
;  which could run in racket by writing "λ" instead of "Lambda":
(λ (condition)
  (λ (consequent)
    (λ (alternative)
      ((condition consequent) alternative))))

#| None of that transformation is computation *in* the Lambda Calculus.

 It's just a *notation*, and we perform a syntactic transformation to determine which
  expression in the Lambda Calculus a person means when they use it.

 Imagine writing a Python program and sometimes writing C style loops.
 You could ask someone, who doesn't know C nor Python, to look for this pattern in your code

   for(init; condition; update) {
       statement ;
       ... }

  and replace it everywhere with

   init
   while condition:
     statement
     ...
     update

  so it can run.

 Or a researcher studying regular [as in CSC236 regular expressions and DFAs] languages
  might use r+ to mean r∘r*. They could then pay someone minimum wage to go through their
  papers and mechanically replace every

   <something>+

  with

   <something>∘<something>*

 And first-order logic uses the notation

   ∃! <id> ∈ <domain>, <expression>

  as shorthand for

   ∃ <id> ∈ <domain>, <expression> ∧ ∀ y ∈ <domain>, <expression>[<id>/y] ⇒ y = <id>

  where [<id>/y] means replace the variable <id> with the variable y.


 When Lisp was created, it was universally understood that a sane general purpose programming
  language would have syntactic abstraction, because that's fundamental to how we express ourselves.
  Everyone still knew that the design of languages implemented to implement programs to run on 1950s
  machines were all about the limitations of 1950s machines and the limited set of programs worth
  running on them. It would have been considered absurd to study those languages to learn how we
  want to express ourselves.

 By the 1970s Jon Backus tried to remind programmers that he made Fortran for the 1950s, and
  programmers should stop using it and its descendants for the vast majority of programs to be
  run on the *exponentially* more capable computers of the 1970s, but it was too late. |#
