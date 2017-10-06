#| The Semantics of Function Call

A Programming Language Principles course cares mainly about how programming languages behave
 [the "semantics", aka the "meaning"], not how they look [the "syntax"].

In Python, C, and Javascript, the CSC 104 language, and Racket: a function call expression
 mentions a function and argument expressions. The syntax is one of:
  • <function-name>(<argument-expression>, ...)
  • (<function-name> <argument-expression> ...)

After removing any function call syntax [parentheses, commas, spaces here], we have:
  • <function-name>
  • <argument-expression>s

Then the semantics is:
  1. Evaluate each <argument-expression>, in order, to produce the argument values.
  2. Evaluate the body of the function, using the argument values as the values of the parameters.

Step 1 is done without using any information about the function.
Which function will be called has absolutely no effect on step 1.

Evaluating *all* the argument expressions is called "eager" or "strict" evaluation.

Notice step 2 is done without using any information about the argument expressions, except
 their values. For example, when the function is activated, it's impossible to distinguish
 whether the values came from simple constants, or an expression hundreds of pages long.

Giving the function only the values of the argument expressions is called "call by value"
 or "pass by value".

Please memorize, and use as appropriate, the terms "eager" [or "strict"], and "call by value"
 [or "pass by value"].

As one student pointed out: Stackoverflow isn't a good place to learn these things, in particular
 he mentioned the endless debates about whether Java method call is by value or not.

[Java method call is by value.

 You can convince yourself now by asking whether it's possible to write method m that can determine
  which of these two ways it was called:
    • o.m(324);
    • o.m(300 + 24);

 If the reference/object distinction is the concern:

   Object o1 = new Object();
   Object o2 = o1;
   // Can m know which variable produced the reference it receives here:
   o.m(o1);
   o.m(o2); ] |#


#| Installing the csc104 Language

 Select the DrRacket menu item "Install Package ...", and enter:

   csc104

  [lower case, no spaces] in the "Package Source" box, and click "Install".

 When it's finished a nice message with some ★s will appear in the output telling you to close
  the dialog box and restart DrRacket.
 When you restart DrRacket open this file again and the CSC 104 language will be selected. |#


#| Function Call in the csc104 Language

 The syntax is:  (<function-name> <argument-expression> ...)  |#

25 ; A numeric literal.
"solid" ; A string literal.
. ; An image literal.

pi ; A variable referring to a number
triangle ; A variable referring to a function.

; Calling a function with the wrong number of arguments, comment it out after running once:
; (triangle)
; Calling a function, with three arguments:
(triangle 25 "solid" "maroon")

(+ 3 2 4) ; Add 3, 2, and 4.
(beside (triangle 25 "solid" "maroon") (triangle 25 "outline" "blue"))
(above . (beside . .))

; Click the "Reindent All" button, or use the keyboard shortcut Control-I or Command-I:
(+ (* 3 100)
   (* 2 10)
   4)

; Click the "Finer Format" button:
(flip-vertical(above .
                     (beside .
                             .)
                     )
              )


#| We have two special forms in the csc104 language to see the steps of evaluation:
    • 'step' which waits for enter/return between steps
    • 'steps' which shows all the steps without waiting |#

(step (beside (triangle 25 "solid" "maroon") (triangle 25 "outline" "blue"))
      (+ (* 3 100)
         (* 2 10)
         4)
      (above . (beside . .)))

; Note carefully each step: argument expressions are reduced to a value before the function is called.


#| Function Definition in the csc104 Language

 The syntax is:

   (define (<function-name> <parameter-name> ...)
     <result-expression>)

 As is typical in programming languages, the "header" (<function-name> <parameter-name> ...)
  is in the same form as a call, except with parameter names instead of argument expressions.

 Less typically: there is exactly one expression for the body. |#

(define (stack an-image)
  (above an-image (beside an-image an-image)))

#| Step 2 of Function Call, in the csc104 Language

 Evaluating the body of a function means: replace the call with the body of the function,
  and replace the parameters with the argument value(s) [from step 1!]: |#

(step (stack .))
(step (stack (stack .)))

#| Without using 'step' or 'steps', is it detectable whether that really is the evaluation rule,
    or the evaluation rule is: substitute the argument *expressions* into the body? |#


#| A comment about comments:

 #| This is a block comment, which unlike in Java, properly nests, i.e.
     this closing delimiter doesn't end the outer one: |#

 ; This is an end-of-line comment

 #;(the "#;" comments out
     the next syntactically correct form but does not color it as a comment) |#


; Back to function call. Is the final result different if
#;(stack (stack .))
;  turned into
#;(above (stack .) (beside (stack .) (stack .)))


; What do the following steps tell you about 'and':
(step (and (= 1 2) (= 3 3)))

; Consider:
(define (And b0 b1)
  (and b0 b1))

; What is essentially different about these steps:
(step (And (= 1 2) (= 3 3)))

; Predict:
#;(and (= 1 2) (/ 1 0))
#;(And (= 1 2) (/ 1 0))

#| Conjunction in Most Programming Languages

 In Python:

   def And(b0, b1): return b0 and b1

   (1 = 2) and (1 / 0)
   And(1 = 2, 1 / 0)

Conjunction in Python, Java, C, Javascript, csc104, and racket is "short-circuiting":
 in particular it isn't strict/eager, so doesn't have function call semantics.

In Python, but not in csc104, the syntax for it is different from calling a function,
 but that's a minor difference. |#


#| Declaring and Initializing Non-Function Variables in csc104

 (define <variable-name> <initialization-expression>) |#

(define favorite-triangle (triangle 27 "solid" "forestgreen"))
favorite-triangle

(step (stack favorite-triangle))

; What if we try to define 'define'?
#;(define (Define variable value)
    (define variable value))

; That's not actually valid: the body expression can't be a 'define'.
; But pretend we made a valid function 'Define'. What must happen here:
#;(Define two (+ 1 1))

; Hint:
#;(+ three three)
#;(stack three)

#| Variable Initialization in Most Programming Languages

 In Python:

   def initialize(variable, value): variable = value

   initialize(two, 1 + 1)

Initialization in Python, Java, C, Javascript, csc104, and racket doesn't evaluate the variable,
 but receives the variable name: it isn't by value, so doesn't have function call semantics.

In Python, but not in csc104, the syntax for it is different from calling a function,
 but that's a minor difference. |#


; If we didn't have [binary] addition, we could define it:
(define (add x y)
  (- (- x (- y))))

#| Addition in Most Programming Languages

 In Python:

   def add(x, y): return - ((- x) - y)

That can be used anywhere we could use '+' [for numbers].

Addition in Python, Java, C, Javascript, csc104, and racket evaluates both expressions,
 and receives only their values: it has function call semantics.

In Python, but not in csc104, the syntax for it is different from calling a function,
 but that's a minor difference. |#


#| Python and: special semantics,       special syntax
   csc104 and: special semantics,       function call syntax
   Python   +: function call semantics, special syntax

In many languages:  special semantics ⇒ special syntax
                     ¬[special syntax ⇒ special semantics]

When people mention a language feature, they often mention it by its syntax, causing confusion.

For example, if someone asks whether a language has Python's 'in': do they mean the semantics
 [a function that determines whether an element is in a list], or a specialized notation for
 that one particular function? |#


(define (f x ignore) (* x x))
; Predict:
#;(f 18 (/ 1 0))

#| Predict:

  def f(x, ignore) : return x * x

  f(18, 1 / 0) |#


#| EXERCISE. |#
; Predict, and try the same experiment in Python:
(define g 1)
#;(g 2)
#;(g (/ 1 0))


#| EXERCISE.

In Java, incrementing a variable has special syntax: ‘x++’ .
Does it have function call semantics?

An attempt to define it in Python:

  def inc(x): x = x + 1

  x = 108
  inc(x)
  print(x) # What does this print?
  y = 108
  inc(y)
  print(y) # What does this print? |#


#| EXERCISE.

Can 'neither' be defined in Python?

  while neither(node == None, node.value == key):
    node = node.next

Give an example that would break the expected behaviour of such a 'neither'.
E.g. why does this definition fail to work:

  def neither(b1, b2): return (not b1) and (not b2) |#
