#| Local Scopes, and Local Functions

 In most languages, including Python, Java, and csc104, function parameters
  are in a local scope: they can't be referenced outside of the body.

 There are various operations in languages that create scopes inside other scopes.
 In csc104, we have an explicit scope creation expression usable anywhere an expression
  can be used: |#

#| Example |#
(local [(define x 18)]
  ; Can use x in the body ...
  (sqr x))
;  ... but not outside the local expression:
#;
(local [(define x 123)]
  (sqr x))

#| General Form |#
#;(local [<definition>
          ...]
    <body-expression>)

#| Relationship to Function Call |#

; The example is *very* similar to making a function with parameter 'x' and body (sqr x) ...
(define (f x)
  (sqr x))
;  ... and immediately calling it with the value of x:
(f 18)

; Is there a *detectable* difference?

; What about in our current underlying model?
; When is each expression evaluated here, including variable reference substitutions:
(step (define (f x)
        (list (sqr x) (sqr x)))
      (f (random 1000))
      (local [(define x (random 1000))]
        (list (sqr x) (sqr x))))

; We will return to this later, expressing various scoping operations explicitly
;  with only the core programming language operations of function creation and
;  function call.


#| "Shadowing" |#

; Local definitions can "shadow" less local definitions:
(local [(define list 123)]
  list)


#| Local Functions

 You are probably less familiar, with making functions in a local scope, especially
  inside the scope of another function's body.

 Object-Orientation is however based on this mechanism, with special syntax and
  terminology being the main difference. We will return to OO later and express
  it explicitly with the core programming language operations.

 The main power of making a function inside another function, besides the usual
  benefits of naming, is having the function's behaviour depend on the arguments
  to the outer function. |#

; Example.

(define database
  (list (list "Alonzo Church" 1903 .)
        (list "Grace Hopper" 1906 .)
        (list "Kurt Goedel" 1906 .)
        (list "Miriam Mann" 1907 .)))

(define year second)

(local [(define (born-after-1906? a-person)
          (> (year a-person) 1906))]
  (filter born-after-1906? database))

#;born-after-1906? ; Not in scope.

(step
 ; Expression (E), mentioned again below:
 (local [(define (born-after-1906? a-person)
           (> (year a-person) 1906))]
   (filter born-after-1906? database)))

(define (people-born-after a-year)
  (local [(define (born-after-a-year? a-person)
            (> (year a-person) a-year))]
    (filter born-after-a-year? database)))

; What does the underlying model do?
; The outer function call is still the same: substitute the argument everywhere
;  into the body, including into the body of the inner function:
(step (people-born-after 1906) ; creates the expression (E) stepped above.
      (people-born-after 1905))
