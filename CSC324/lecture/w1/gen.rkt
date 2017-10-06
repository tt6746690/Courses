; define an image
(define t (triangle 15 "solid" "red"))
(image? t)    ; #true

; stack is a function
; upon calling, an-image is substituted with the argument to function
(define (stack an-image)
  (above an-image (beside an-image an-image)))


(stack t)

; doesnt work... cannot make custom define function
; because function call behaves the same way in every circumstances
; evaluates argument expression, then calls the function with values!

; (define (our-define x value)
;  (define x value))
; define: found a definition that is not at the top level


; and
; is not a function
; has special meaning... since prevent second from evaluated if first is false
(and (= 1 2) (= 3 3))
;★ (and (= 1 2) (= 3 3))
;• (and #false (= 3 3))
;• #false

; my-and
; is a function
; does not have special meaning,
; since both are evaluated before substitution into function
(define (my-and v1 v2)
  (and v1 v2))
(my-and #false #true)
; ★ (my-and #false #true)
; • #false


; OK, SINCE short-circuiting comes into play
(steps
   (and ( = 1 2) (/ 1 0)))
; ★ (and (= 1 2) (/ 1 0))
; • (and #false (/ 1 0))
; • #false

; ERROR! will fail to substitute since second expression fails (denominator cant be 0)
(steps
 (my-and (= 1 2) (/ 1 0)))
; ★ (my-and (= 1 2) (/ 1 0))
; • (my-and #false (/ 1 0))
; • /: division by zero


; getaway: 
; special syntaxes, often they signal special meaning as well
;    special sytax doesnt always signal special meaning
; usually has to do with controls as to when expressions are evaluate



; both are equivalent; plus can be re-written as a function 
; plus
(+ 1 2 3 4 5)  ; 15
(- 3 2)        ; 1

; binary plus
; (define (my-plus x y)
;  (- (-(x) -(y))))

; python
; and: special semantics, special syntax
; + : function call semantics, special syntax
; in: function call semantics, special syntax


; list
; a function, evaluates the expression, returns a list
; in the end, shows the list ...
(list 123 "hello" (star 25 "outline" "blue"))
;(list 123 "hello" .)
; note: no memory model...
; consequence:
;    cannot distinguish different lists
;    good: eliminate some of the problems otherwise ...


; identical
(list (random 10) (random 10))    ; (list 0 7)
(list 0 7)                        ; (list 0 7)


(sqr 3)                           ; 9

; map list -> list
; a function that takes
;     a function
;     a list
; apply function to each element in list
(steps (map sqr (list (random 10) (random 10))))

; ★ (map sqr (list (random 10) (random 10)))
; • (map sqr (list 3 (random 10)))
; • (map sqr (list 3 6))
; • (list (sqr 3) (sqr 6))
; • (list 9 (sqr 6))
; • (list 9 36)


(steps (map - (list (random 10) (random 10))))

;★ (map - (list (random 10) (random 10)))
;• (map - (list 4 (random 10)))
;• (map - (list 4 2))
;• (list (- 4) (- 2))
;• (list -4 (- 2))
;• (list -4 -2)


; idea of mapreduce
; if transformation does work independently of each other
; can be parallized easily since no side-effects (mutation)
; the move to stateless programming...

(steps parallel
       (map - (list (random 10) (random 10))))


; ★ (map - (list (random 10) (random 10)))
; • (map - (list 6 7))
; • (list (- 6) (- 7))
; • (list -6 -7)


; side-effect
;  when one expression does something, its detectable with another expression


(define (oval angle)
  (rotate angle (ellipse 10 20 "outline" "red")))
(map oval (list 0 30 60))
; (list . . .)

; range begin end step -> list 
(range 0 10 1)
; (list 0 1 2 3 4 5 6 7 8 9)

(map oval (range 0 90 5))
;(list . . . . . . . . . . . . . . . . . .)


; Note + does no overloads
;   (+ (list 3 2 4))
; +: expects a number, but received (list 3 2 4)



; apply func list -> retTypeOfFunc
; apply list as argument to a function
(steps parallel
       (apply + (list 3 2 4)))
; ★ (apply + (list 3 2 4))
; • (+ 3 2 4)
; • 9

(apply overlay
       (map oval (range 0 90 5)))
(scale 20 (apply overlay
                 (map oval (range 0 90 5))))

; .



; referential transparency
; An expression is said to be referentially transparent if it can be replaced
; with its corresponding value without changing the program's behavior.
; therefore -> deterministic, same output 

; random has side-effect
; violates referential transparency

; note this define fails because function has to have arguments
; since a function with no argument is equivalent to
; a constant with value as output of its body evaluated
(steps  (define r (random 10))
        (r)
        (r))
; define: expects at least one variable after the function name, but found none


; the function r cannot be replaced with the value, as the program behavior might change depending on random
; breaks referential transparency
(define (r _) (random 10))
(steps (r "whatever"))
; 6
(steps (r "whatever"))
; 0


; 3 triangle same
(define (stack-1 i)
  (above i (beside i i)))
; bottom 2 triangle same 
(define (stack-2 i1 i2)
  (above i1 (beside i2 i2)))
  

(define (S-1 n)
  (cond [(zero? n) (triangle 10 "solid" (map random
                                             (list 101 101 101)))]
        [else (stack-1 (S-1 (- n 1)) )]
   )
)


; random violates referential opaqueness
; case 1: since arguments are evaluated before function,
; and that one color dictates the color of stacked triangle
; color is same for all triangles
(S-1 0)
(S-1 1)
(S-1 2)
(steps parallel
       (hide (S-1 1))
       (S-1 2))
(S-1 5)


(define (S-2 n)
  (cond [(zero? n) (triangle 10 "solid" (map random
                                             (list 101 101 101)))]
        [else (stack-2 (S-2 (- n 1)) (S-2 (- n 1)) )]
   )
)

; case 2 : since arguments evaluated before a function
; and the color of bottom 2 and top 1 triangle may be different with random
; but the output has the property that
;     every level is a triangle of same color
(S-2 0)
(S-2 1)
(S-2 2)
(steps parallel
       (hide (S-2 1))
       (S-2 2))
(S-2 5)



