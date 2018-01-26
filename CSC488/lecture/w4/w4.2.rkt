#lang racket



#| Tutorial |#
#; ((λA (x) x) 1)
; put a closure λ0 referencing λA and environment into the result
; call
;   new env E1 under λ0's (from result stack) environment E0
;   put result 1 into E1
;   push current env to call stack
;   set env to E1
;   call λ0
;   set env to E0, the pop of the call stack, when finished evaluating λ0


; closure λA: variable 0
; call λ0: put variable 0 environments up from current env E1 into the result

; λ0 vs λA
; λ0 is the closure, point to the compiled code λA and env E0
; λA is the compiled code



; env:          store pointer to a env
; result:       store pointer to a closure
; result stack: store pointer to closures
; call stack:   store pointer to envs
#; ((λA (a) a) (λb (b) 123))
; put a closure λ0 (λα ·) into result
; push (to) result (stack)
; put a closure λ1 (λβ ·) into result
; push (to) result (stack)
; call
;   - E0 (· λ1) i.e. the parent env and the variable
;   - push env · to call stack
;   - set env to E0
;   - call λ0, look at λα compiled code, i.e. variable 0 in E0 is λ1, push λ1 to result
;   - restore ·


#; (let ([n 0] [s 0])  ; n,s not in scope of initialization expression, so not recursive
     body)
#; ((λ (n)
      (λ (s)
        body)
      0)
    0)

#;(while condition
         body
         ⋯)
#;(letrec ([loop (λ () (if (< n 10)
                           (loop (set! n (+ n 1))
                                 (set! s (+ s n))
                                 (loop))
                           0))])
    (loop))

; you can refer to name loop in init expr
#; (letrec ([loop init])
     body)
#; (let ([loop 0])    ; put in scope, then initialize them
     (set! loop init)
     (body))


; evaluate expression in order
#; (block e0 e1 ...)
#; (let [(_ e0)]
     (block e1 ...))


; 0 arity funtion call
#; (λ () body) ; rewrite to (λ (_) body)
#; (e) ; rewrite to (e 0)







#| Lecture 2 |#

(require (only-in racket [+ racket:+] [define racket:define]))

; automatic currying like Haskell
(define-syntax define
  (syntax-rules ()
    [(define (f-id id) body) (racket:define f-id (λ (id) body))]
    [(define (f-id id ... id-last) body)
     (define (f-id id ...) (λ (id-last) body))]))




#; (define (+ x) (λ y (racket:+ x y)))
#; (define ((+ x) y) (racket:+ x y)) ; curried define

; short-hand
#; (define + (λ (x) (λ (y) (racket:+ x y))))

(define (+ x y) (racket:+ x y))


#; (define (header id ...) body)
#; (define header (λ (id ...) body))

#; (define (+ x) (λ y (racket:+ x y)))
#; (define + (λ (x) (λ (y) (racket:+ x y))))

#; ((if (< n 10)
        (block (set! n ((+ n) 1))
               (set! s ((+ s) n))
               (loop 0)))  ; 0 as dummy variable to function with no args
    0)

; 3 args eager evaluation is a problem
; need delayed evaluation by putting exprs into a function

#; (if ((< n) 10)
       (λ () (block (set! n ((+ n) 1))
                    (set! s ((+ s) n))
                    (loop 0)))
       (λ () 0))

(define (true consequent alternative)
  (consequent 0))

(define (false consequent alternative)
  (alternative 0))

(true (λ (_) 1) (λ (_) (/ 1 0)))   ; output 1
(false (λ (_) (/ 1 0)) (λ (_) 1))  ; output 1


(define (if condition consequent alternative)
  (condition consequent alternative))

(if true (λ (_) 1) (λ (_) 2))   ; 1 
(if false (λ (_) 1) (λ (_) 2))  ; 2


(require (only-in racket [< racket:<] [if racket:if]))

(define ((< x) y) (racket:if (racket:< x y) true false))







