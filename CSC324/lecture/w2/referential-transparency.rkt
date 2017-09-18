;; The first three lines of this file were inserted by DrRacket. They record metadata
;; about the language level of this file in a form that our tools can easily process.
#reader(lib "2017-fall-reader.rkt" "csc104")((modname referential-transparency) (compthink-settings #hash((prefix-types? . #f))))
#| Referential Transparency

 An expression is "referentially transparent" if evaluating it is indistinguishable from just
  using its result value.

 A function is considered referentially transparent if the result of calling it is completely
  determined by the arguments. |#

; So 'random' is not referentially transparent:
(random 324)
(random 324) ; If it depended only on the argument values, this would produce the same result.

; And an expression that meaningfully uses 'random' is not referentially transparent.
; The 'triangle' function can take a list of three red green blue percentages for its color:
(define t (triangle 25 "solid" (map random (list 50 75 100))))
; That recorded the value of (triangle 25 "solid" (map random (list 50 75 100)))).
t   ; t always the same color, since initialization expression is evaluated once
t
t
; Variable access in this language is referentially transparent, not just here but in general:
;  there are no variable update operations, so [in this scope] t always refers to the same value.

(triangle 25 "solid" (map random (list 50 75 100))) ; Probably not equivalent to using earlier value.
; change color on every call

; That is a detectable difference between a language that evaluates the initialization expresssion
;  when the definition is encountered, as opposed to re-evaluating or substituting it each time
;  it's used. Delaying evaluation is a crucial aspect of function definition.

; This language prevents defining user-defined functions that take no arguments, because, except for
;  the special case of randomness, they should be replaced with non-function constant definitions.
#;(define (T) (triangle 25 "solid" (map random (list 50 75 100))))
; define: expects at least one variable after the function name, but found none

; this is because function with no arguments should yield the same result,
; same as just evaluating initialization expr once for constants

(define (T _) ; Put in an unused parameter.
  (triangle 25 "solid" (map random (list 50 75 100))))
(T "whatever")
(T "whatever") ; Probably different: T is not referentially transparent.


(define (stack an-image)
  (above an-image
         (beside an-image an-image)))

; Conditional evaluation in the language:
#;(cond [<condition-expression> <consequent-expression>]
        [else <alternative-expression>])

; not referentially transparent, since using random here..
(define (S n)
  (cond [(= n 0) #;(zero? n) (triangle 25 "solid" (map random (list 50 75 101)))]
        [else (stack (S (sub1 n) #;(- n 1)))]))

(step parallel
      (S 0)
      (S 0)
      (S 1))

; There's an option for step[s] to hide particular function calls:
(step [hide (S 1)]
      (S 2))

; What if we replace the call to stack with the body of stack, substituting the expression that
;  was producing the argument:
(define (S-III n)
  (cond [(zero? n) (triangle 25 "solid" (map random (list 50 75 101)))]
        [else (above (S-III (sub1 n))
                     (beside (S-III (sub1 n)) (S-III (sub1 n))))]))

(step [hide (S-III 0)]  ;  will have 3 different colors...
      (S-III 1))

; That difference demonstrates that function call is not substituting the argument expressions.


(define (stack-II an-image another-image)
  (above an-image
         (beside another-image another-image)))

(define (S-II n)
  (cond [(zero? n) (triangle 25 "solid" (map random (list 50 75 101)))]
        [else (stack-II (S-II (sub1 n)) (S-II (sub1 n)))]))

(step [hide (S-II 0)]    ; each level has same color... pattern: bot 2  triangle same color differnt from the top one
      (S-II 1))
(step [hide (S-II 1)]
      (S-II 2))

; Exercise: ponder the difference in execution between these two ways of adding up 0 ... n:
(define (add-up-to n)
  (cond [(positive? n) (+ (add-up-to (sub1 n)) n)]
        [else 0]))
(define (Add-up-to n sum-so-far)
  (cond [(positive? n) (Add-up-to (sub1 n) (+ sum-so-far n))]
        [else sum-so-far]))
(step parallel
      (add-up-to 5)       ; (+ (+ (+ (+ (+ 0 1) 2) 3) 4) 5)
      (Add-up-to 5 0))    ; just use one varaible sum-so-far...
; What's the Big-Θ usage of memory for each of them?

; add-up-to O(n)   (recursion) 
; Add-up-to O(1)   (iteration)

