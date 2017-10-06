#| Type Predicates

 We've seen five types:

   • image
   • number
   • string
   • boolean
   • function

 There are corresponding "unary" [taking one argument] "predicates" [functions producing a boolean] |#

(number? .)

(number? "324")

(string? "324")

(image? .)

(boolean? (image? .))

function?

(function? number?)

(function? (number? "324"))


#| Lists: Another Datatype |#

list ; The list function.
(list (sqr 18) "hey" (circle 50 "outline" "red"))

; This language has no print statement, no toString/repr/str, and no separate "memory model":
;  evaluation is by "rewrite rules", also known as algebraic reduction, "reducing" expressions
;  to simplest canonical form. Values are shown as simplest expressions reproducing the value:
(list 324 "hey" .)


#| Mapping

 Processing every element of a list, independently "in parallel", is a common operation.

 Having it explicit, as opposed to examining whether a loop has that form, has two related benefits:

   The reader can see what's going on more clearly.

   The machine [compilers, program verifiers and testers, etc] can see what's going on more clearly.
     So execution can be optimized, for example using multiple cores or machines, or the GPU which
      has a specialized architecture that is often optimal for executing maps versus iterations.
     At least one of our research faculty members works on tools to reverse-engineer the intent
      of loops in legacy code. |#

(step (map string? (list (sqr 18) "hey" (circle 50 "outline" "red"))))

#;(map (string?) (list (sqr 18) "hey" (circle 50 "outline" "red"))) ; What error does this generate?
; string?: expects one argument, but found none

#| The algebraic rewrite rule for map is: (map f (list a b c ...)) → (list (f a) (f b) (f c) ...) . |#


#| The 'parallel' option for step[s] reduces leaf terms of nested function calls in parallel. |#

(step parallel
      (map string? (list (sqr 18) "hey" (circle 50 "outline" "red"))))

; But 'and' is still special:
(step parallel
      (and (< 2 1) (= 3 3))
      (and (= 1 1) (< 2 1) (= 3 3)))

; A couple more examples of mapping:
(step parallel
      (map - (list 3 2 4))
      (map sqr (list (random 10) (random 10))))

; And with a user-defined function:
(define (oval angle)
  (rotate angle (ellipse 10 20 "outline" "red")))
(step parallel
      (map oval (list 0 30 60)))

; Function 'range':
(range 0 90 30) ; List of numbers from 0 to 90, stepping by 30, not including 90.
(map oval (range 0 90 5)) ; List of lots of ovals.


#| Applying

 Combining the elements of a list.

 The rewrite rule for 'apply' is: (apply f (list a b c ...)) → (f a b c ...) .

 Early in Google's history they had a MapReduce framework for distributed execution of problems
  that were expressed as maps followed by a "reduce" to combine the results [we'll discuss the
  specifics of when a combination is a "reduce" later, but all the variadic functions use here
  do a "reduce"]. As with mapping, there is research into reverse-engineering loops to determine
  if they are "reduce"s, because they are partially parallelizable [to be discussed later]. |#

#;(+ (list 3 2 4)) ; What error does this generate?
;+: expects a number, but received (list 3 2 4)

(step (apply + (list 3 2 4))) ; Use a list of arguments as the arguments.
; Is there a difference if we use the parallel option: No! same thing
(step parallel
      (apply + (list 3 2 4)))
; Contrast with:
(step parallel
      (map + (list 3 2 4)))  ; parallel works, but returns a list instead

#;(apply - (list 3 2 4)) ; What error does this generate?
; -: expects only 1 to 2 arguments, but found 3

; Two more to compare:
(step parallel
      (apply list (list 3 2 4))      ; (list 3 2 4)
      (map   list (list 3 2 4)))     ; (list (list 3) (list 2) (list 4))
 
; Function 'overlay' takes any number of images and overlays the first on the second on ..., centered.
; Compare:
(step parallel
      (apply overlay      
             (list (triangle 30 "solid" "green")     ; does the overlay 
                   (star 25 "outline" "blue")
                   (square 30 "solid" "red")))
      (map overlay
           (list (triangle 30 "solid" "green")      ; a list of images, no overlay
                 (star 25 "outline" "blue")
                 (square 30 "solid" "red"))))

; Let's have some fun:
(scale 30 (apply overlay (map oval (range 0 90 5))))

; Someone asked about more variadic functions:
*
beside
above
string-append

(step parallel
      (apply string-append (list "hello" " " "world!"))
      (map string-append (list "hello" " " "world!")))

; Exercise.
#;(apply _ (list 324)) ; Fill in the blank to make this not an error.
#;(apply star _) ; Fill in the blank to make this not an error.
