#lang racket

#| The Key Benefit of Functional Programming.

 Students asked about this again in the night section, and afterwards I remembered a nice example
  from debugging a student's Python code a few years ago. If you were in today's day section lecture
  you saw this example, but you will probably find the new information in the comment at the very end
  interesting.

 We cover functional programming in CSC324 because FP is becoming ubiquitous in modern programming,
  and of the common programming patterns it's the least covered by other courses. But there's a good
  reason it's becoming ubiquitous. If we define functional programming as “stateless” programming,
  then the key benefit is: it's easier to reason about, by us and tools.

 Tools: verifiers, optimizing compilers, parallelizing compilers, etc.
 Easier for us to reason about: making correct, maintaining, optimizing [even if the initial version
  of a program has sub-optimal performance], etc. |#

; People often associate FP with recursion: for repetition, possibly through higher order functions
;  such as map, fold, filter, apply, etc. So let's remove that distraction, by making a traditional
;  stateful for loop, and a stateless version.

(define-syntax-rule (For! (id bound)
                          body ...)
  (local [(define id 0)
          (define (loop)
            (when (< id bound)
              body ...
              (set! id (add1 id))
              (loop)))]
    (loop)))

(define-syntax-rule (For (id bound)
                         body ...)
  (local [(define (loop id)
            (when (< id bound)
              body ...
              (loop (add1 id))))]
    (loop 0)))

(define sum! 0)
(For! [n 10] (set! sum! (+ sum! n)))

(define sum 0)
(For [n 10] (set! sum (+ sum n)))

(require rackunit)
(check-equal? sum! sum) ; So far, no difference.

; Most programming languages are high level enough that the effects of different styles are hard to
;  see with just a few lines, or even a few pages, of code. And the specification of the program
;  usually needs to be sufficiently complex to justify pages of code.

; But let's look at a common approach to event handling in GUI and Web programming, where state first
;  confounds many programmers. A student asked me to debug the situation below in some Python code,
;  but there's a punchline at the end from a search for this in the Javascript world [where “callback”
;  style programming is common].

(require racket/gui)
; Makes two window objects, details unimportant.
(define window  (new frame% [label "Functional"] [width 320]))
(define window! (new frame% [label "Stateful"] [width 320]))

; Add two buttons to window:
(For [n 2] (new button%
                [label (number->string n)]
                [parent window]
                ; Click handler: display the button's number in the Interactions.
                [callback (λ args (displayln n))]))
; Add two buttons to window!:
(For! [n 2] (new button%
                 [label (number->string n)]
                 [parent window!]
                 ; Click handler: display the button's number in the Interactions.
                 [callback (λ args (displayln n))]))
(send window show #true)
(send window! show #true)
; Click the buttons: what happens?

; This is a frequent source of bugs and confusion in Javascript programming:
;  https://stackoverflow.com/questions/750486/javascript-closure-inside-loops-simple-practical-example
;
; More significantly: Javascript recently added local variable declarations, and using them is now
;  considered “best practice”. In particular, local loop variables behave functionally!
;
; See §2.1 and §4 of: http://2ality.com/2015/02/es6-scoping.html
