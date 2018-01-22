#lang racket

; (require (rename-in "tree.rkt" (s-expression→tree-picture tree)))
; (current-font-size 20)

; We'll view nested lists as trees
;     List is an internal node.
;       First element of a list is the label of an internals
;       Rest of the elements are the children

#; (tree 'a)
; Non-list: draw it as is
; List: draw a circle put the first element in it,
;   put the drawings of the children underneath

#; (tree '(a))     ; a
#; (tree '(a b))   ; circle of a, underneath it b
#; (tree '(a b c)) ; circle of a, underneath it b, c
#; (tree '(a (b))) ; circle of a, underneath circle of b
#; (tree '())      ; empty circle


; Let's enrich our language
; Extended LC -> LC -> x86

#; (tree '(λ (a) (let (b c) (d e))))
;     λ
;   a   let
;      b   d
;      c   e

; Transform let to a λ
#; (tree '(let (b c) (d e)))
#; (tree '((λ (b) (d e)) c))
#; (tree '(#%app (λ (b) (#%app d e)) c))  ; make function call explicit




; tree transformation
(define (expand-let v)
  (match v
    [`(let (,id ,init) ,body) `((λ (,id) ,body) ,init)]
    [_ v]))

; tree traversal
(define (expand v)
  (match v
    [`(let ,_ ,_) (expand (expand-let v))]
    [`(,e1 ,e2) `(,(expand e1) ,(expand e2))]
    [`(λ (,id) ,body) `(λ (,id) ,(expand body))]
    [_ v]))



(require rackunit)
(check-equal? (expand-let 'a)
              'a)
(check-equal? (expand-let '(let (b c) d))
              '((λ (b) d) c))
(check-equal? (expand-let '(let (b c) (d e)))
              '((λ (b) (d e)) c))

(check-equal? (match '(1 2 3)
                [(list a 4 c) (+ a c)]
                [(list a 3 c) (* a c)]
                ['(a b c) 'literal-identifier]
                ['(1 2 2) 'literal]
                [`(1 2 4) 'quasi]
                [`(1 ,x 3) x])
              2)

(check-equal? (match '(let (b c) d)
                [`(let (,id ,init) ,body) (list id init body)])
              '(b c d))

(check-equal? (local [(define id 'b)
                      (define init 'c)
                      (define body 'd)]
                `((λ (,id) ,body) ,init))
              '((λ (b) d) c))

(check-equal? '((λ (b) d) c)
              (match '(let (b c) d)
                [`(let (,id ,init) ,body) `((λ (,id) ,body) ,init)]))

(check-equal? (expand '(let (b c) d))
              (expand-let '(let (b c) d)))
(check-equal? (expand '((let (b c) d) x))
              `(,(expand-let '(let (b c) d))
                x))
(check-equal? (expand '(let (b (let (x y)
                                 z))
                         d))
              (expand (expand-let '(let (b (let (x y)
                                             z))
                                     d))))



(check-equal? (expand '(let (fib (λ (n) (fib n)))
                         (fib 10)))
              '((λ (fib) (fib 10)) (λ (n) (fib n))))
; fib not defined


#; (let (fib 0)
     (set! fib (λ (n) (fib n)))
     (fib 10))

; recursion; put a pointer in λ's env to refer to itself


