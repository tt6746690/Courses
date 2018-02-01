#lang racket

; Let's enrich our language of the Lambda Calculus, as part of the compilation chain:
;   Syntax → parse tree → enriched LC → LC → x86.

; A little library to draw nested lists as trees.
(require "tree.rkt")
(current-font-size 12)

#| We'll view nested lists as ordered trees.

 There are two common interpretations, we'll mainly use this one:
   A list is an internal node, with the first [“head”] element as the label/value/key/data,
    and the rest [“tail”] of the elements as children.

 Drawing:
   • a list
     - draw a circle
     - put the first element in the circle [if there is a first element]
     - put the drawings of rest of the elements underneath in order, connecting to them with edges
   • non-list
     - draw as-is |#

#;(tree 'a)
#;(tree '(a))
#;(tree '(a b))
#;(tree '(a b c))
#;(tree '(a (b)))
#;(tree '(a (b c) d))
#;(tree '())

; We'll extend the LC with a ‘let’ form:
#;'(let (b (c d)) (b e)) ; let b be the value of (c d), and evaluate (b e) accordingly.
; means
#;'((λ (b) (b e))
    (c d))

; Notice the two labels ‘b’ mean different things: one is a declaration, and the other is the
;  expression in a function call. In particular, the label isn't enough to determine the
;  meaning of the children. We'll improve that later.

; First, let's make a function ‘expand-let’ to transform an expression if it's a let form.

; This makes a “sub-module”, that runs when this file is run, but *after* the rest of the file runs.
;  In particular, it can use ‘expand-let’ even though that's defined later.
(module+ test

  ; ‘rackunit’ is a unit testing library, in particular it has ‘check-equal?’.
  (require rackunit)

  ; The earlier example.
  ; Too complicated as part of systematic design and testing!
  (check-equal? (expand-let '(let (b (c d)) (b e)))
                '((λ (b) (b e))
                  (c d)))

  ; We'll leave other forms alone.
  (check-equal? (expand-let 'a)
                'a)
  ; Simplest possibile.
  (check-equal? (expand-let '(let (b c) d))
                '((λ (b) d) c))

  ; Our review of pattern matching.
  (define (m v)
    (match v
      [(list a 4 c) (+ a c)]
      [(list a 3 c) (* a c)]
      ['(a b c) 'literal-identifiers]
      ['(1 2 2) 'literal]
      [`(1 2 4) 'quasi]
      [`(1 ,x 3) x]))
  (check-equal? (m '(1 4 3)) 4)
  (check-equal? (m '(1 3 3)) 3)
  (check-equal? (m '(a b c)) 'literal-identifiers)
  (check-equal? (m '(1 2 2)) 'literal)
  (check-equal? (m '(1 2 4)) 'quasi)
  (check-equal? (m '(1 5 3)) 5)

  ; Design: does our pattern extract the components we expect?
  (check-equal? (match '(let (b c) d)
                  [`(let (,id ,init) ,body) (list id init body)])
                (list 'b 'c 'd))
  ; Design: do our template for the result produce the expected result?
  (check-equal? (local [(define id 'b)
                        (define init 'c)
                        (define body 'd)]
                  `((λ (,id) ,body) ,init))
                '((λ (b) d) c))

  ; Design: does combining those produce the expected result?
  (check-equal? '((λ (b) d) c)
                (match '(let (b c) d)
                  [`(let (,id ,init) ,body) `((λ (,id) ,body) ,init)]))

  ; Design: the function should produce the same result as that.
  ; And examining it, we think it's a full design: not dependent on the particular argument value.
  (check-equal? (expand-let '(let (b c) d))
                (match '(let (b c) d)
                  [`(let (,id ,init) ,body) `((λ (,id) ,body) ,init)])))


; Copy [literally, to avoid typos, don't retype it!] that last test,
;  replace ‘check-equal?’ with ‘define’,
;  choose a parameter name and replace the example argument consistently:
(define (expand-let v)
  (match v
    [`(let (,id ,init) ,body) `((λ (,id) ,body) ,init)]
    [_ v]))

; We should consider more tests, but the above is a helper for ‘expand’, and a lot of its
;  design and testing will explicitly [not just implicitly] test ‘expand-let’.

(module+ test ; Extends the earlier sub-module.

  ; Behave like ‘expand-let’ if it's a let form without any let forms inside.
  (check-equal? (expand '(let (b c) d))
                '((λ (b) d) c))
  ;
  (check-equal? (expand '(let (b c) d))
                (expand-let '(let (b c) d)))

  ; Next simplest example that can't be treated the same as any previous.
  (check-equal? (expand '((let (b c) d)
                          x))
                '(((λ (b) d) c)
                  x))
  (check-equal? (expand '((let (b c) d)
                          x))
                `(,(expand-let '(let (b c) d))
                  x))
  
  ; Next simplest example that can't be treated the same as any previous.
  ; Design only:
  (check-equal? (expand '(let (b (let (x y)
                                   z))
                           d))
                (expand (expand-let '(let (b (let (x y)
                                               z))
                                       d))))

  ; Exercise: flesh out the design and testing systematically.
  )

(define (expand v)
  (match v
    
    [; The underscore pattern matches anything, multiple uses don't need to be the same,
     ;  it can't be used in the result.
     `(let ,_ ,_)
     ; Notice the per-order/top-down processing: ‘let’ affects the meaning of the children, before
     ;  continuing to expand.
     (expand (expand-let v))]

    ; Notice: includes knowledge that the first child of λ is not an expression (the args ...).
    [`(,e1 ,e2) `(,(expand e1) ,(expand e2))]
    [`(λ (,id) ,body) `(λ (,id) ,(expand body))]
    
    [_ v]))
