#lang typed/racket


(require/typed typed/racket
               (first (∀ (α) (Listof α) → α))
               (rest (∀ (α) (Listof α) → (Listof α)))
               (map (∀ (α β) (α → β) (Listof α) → (Listof β)))
               (list (∀ (α) (α * → (Listof α)))))
               
(define-type Stack (Listof Real))
(struct (α) stack-result ((stack : Stack) (result : α)) #:transparent)


; lift takes a function, and transform the function such that
; the new function takes in a stack and return a corresponding stack-result whose result is result of the function
; Note lift's type is parameterized over parameter and return type of the input function
(: lift : (∀ (α β) ((α → β) → (α → (Stack → (stack-result β))))))
(define (((lift f) a) s)  ; curried
  (stack-result s (f a)))


; part a
(: number-of-items : (Stack → (stack-result Natural)))
(define (number-of-items stack)
  (match stack
    [`(,e ...) (stack-result stack (length e))]))  ; idea is use match to convert Stack → list, s.t. you can use length on it

(number-of-items '(1 2 3 4 5 6))
;(stack-result '(1 2 3 4 5 6) 6)

; part b
(: lift′ : (∀ (α β) ((α → β) → (α → (Stack → (stack-result β))))))
(define lift′
  (λ (f)
    (λ (a)
      (λ (s)
        (stack-result s (f a))))))


; part c
(ann println (Any -> Void))
(: g : (Any → (Stack → (stack-result Void))))
(define g (lift println))

; part d
; compile time assertion for a call of g
(ann (g 1) (Stack → (stack-result Void)))
; runtime assertionof a call of g
(: g-call : (Stack → (stack-result Void)))
(define g-call (g 1))



; part e
((g 'calling-g) '(0 1 2 3))
; (stack-result '(0 1 2 3) #<void>)
; side-effect: 'calling-g symbol is printed