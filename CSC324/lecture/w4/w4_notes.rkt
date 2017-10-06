;; The first three lines of this file were inserted by DrRacket. They record metadata
;; about the language level of this file in a form that our tools can easily process.
#reader(lib "htdp-intermediate-lambda-reader.ss" "lang")((modname w4_notes) (read-case-sensitive #t) (teachpacks ()) (htdp-settings #(#t constructor repeating-decimal #f #t none #f () #f)))

#| anonymous functions

    e.g. python, js, ruby, kotlin, lisp/scheme/racket,
         haskell, *modern* java, *modern c++*, objecive C, swift
|#

#| components of a function

 + parameter names
 + body expression, which can be "open" in the parameter names
    aka, the parameter names can be "free"/"unbound"

(lambda (n) (* 2 n))
|#

; (define (... n) (* 2 n))

(λ (n) (* 2 n))
((λ (n) (* 2 n))
 100)


; a literal for a function, much same like #true for boolean
(define double (λ (n) (* 2 n)))

; like returning a local function
(local [(define (double n) (* 2 n))]
  double)


; converts first form to second form under the hood 
#; (define (<function-identifier> <parameter-id> ...)
     <body-expression>)
#; (define <function-identifier>
     (λ (<parameter-id> ...)
       <body-expression>))


; Fortran
; a layer between a small domain of programs which can be translated to machine code
; then he wrote a paper for stopped using Fortran since its its too low level



; usage: one-time function
(map (λ (n) (* 2 n)) (list 3 2 4))


#; (* 2 n)    ; open in n

(define (fix-1st f 1st)
  #;(local [(define (f′ 2nd)
              (f 1st 2nd))]
      f′)
  (λ (2nd) (f 1st 2nd)))

(define (fix-2nd f 2nd)
  (λ (1st) (f 1st 2nd)))

(define >100 (fix-1st > 100))
(filter (fix-1st > 100) (list 324 123 56))


#;(define (¬ p)
  (λ (v) (not (p v))))

(define (∘ f g)
  (λ (v) (f (g v))))


(define (¬ p)
  (∘ not p))


;db
(define database
  (list (list "ada" 1815)
        (list "alan" 1912)))

(define birth second)

(filter (λ (person) (> (birth person) 1905))
        database)

(define p (fix-2nd > 1905))

; free point like
; functions are unary
(filter (∘ p birth)
        database)


(check-expect (parts (list 1 (list 2 3) 4))
              (list* (list 1 (list 2 3) 4)
                    
                    (list
                     1
                     (list 2 3)
                     2
                     3
                     4)))

(check-expect (parts (list 1 (list 2 3) 4))
              (list* (list 1 (list 2 3) 4)
                    
                    (append (list 1)
                    
                            (list 
                                  (list 2 3)
                                  2
                                  3
                                  4))))


#;(check-expect (parts (list 1 (list 2 3) 4))
              (list* (list 1 (list 2 3) 4)
                    
                    (append (list 1)
                    
                            (parts (list (list 2 3) 4)))))


#;(check-expect (parts (list 1 (list 2 3) 4))
              (list* (list 1 (list 2 3) 4)
                    
                    (append (parts 1)
                    
                            (parts (list (list 2 3) 4)))))



#;(check-expect (parts (list 1 (list 2 3) 4))
              (list* (list 1 (list 2 3) 4)
                     (apply append (map parts (list 1
                                                    (list (list 2 3) 4))))))

(define (parts v)
  (list* v (cond [(list? v) (apply append (map parts v))]
               [else (list)])))




; if first element of list is equal to "tag"
; make-tag? is a class that takes a tag
; returns a function that matches the tag
(define (make-tag? tag) (∘ (fix-2nd equal? tag) first))
(define href? (make-tag? 'href))

; (filter (make-tag? 'img) ne)



#|

Symbols

     1905s and later ...
     Strings were expensive

     Symbols were datatype for names where you care only
     about identity, i.e. equal?
     for creating names

|#

(quote abc)
; 'abc

(define (q name) (quote name))
#; (q abc)
; abc: this variable is not defined
(q 123)
; 'name



(quote (1 2 3))
; (list 1 2 3)

'(1 2 3)
; (list 1  2 3)


; same
(quote (peter matt))
(list (quote peter) (quote matt))
; (list 'peter 'matt)


(list (quote define)
      (quote <function-identifier>)
      (quote (λ (<parameter-id> ...)
               <body-expression>)))


(list (quote define)
      (quote <function-identifier>)
      (list (quote λ)
            (quote (<parameter-id> ...))
            (quote <body-expression>)))

#|
 (list
 'define
 '<function-identifier>
 (list 'λ (list '<parameter-id> '...) '<body-expression>)) |#


; easy to scan the components...
; for compilers interpreters...



#| Little data language of Calculus functions

   By structural induction/recursion, an expression is:
   cos
   sin
   (<f> + <g>), where <f> and <g> are expressions
   (<f> · <g>), where <f> and <g> are expressions
   (- <f>),     where <f> is an expression
|#


'cos
'sin
'(cos + sin)          ;equivalent to (list 'cos '+ 'sin)
'((cos + sin) · sin)
'(- (cos + sin))



; match not a function, an operation
(match 123
  [100 "hundred"]
  [123 "one hundred twenty three"])
; "one hundred twenty three"

(match (list)
  [(list) "empty"]
  [123 "one hundred twenty three"])
;  "empty"

(match (list 123)
  [(list x) x]
  [123 "one hundred twenty three"])
;  123


(match (list 567 123)
  [(list x 123) x]
  [123 "one hundred twenty three"])
; 567


(match '(cos + sin)
  [(list <f> '+ <g>) <g>])  ; <f> are variable that matches to cos
; 'sin


(define (δ f)
  (match f
    ['cos '(- sin)]
    ['sin 'cos]
    [(list <f> '+ <g>)
     (list (δ <f>) '+ (δ <g>))]
    [(list '- <f>) (list '- (δ <f>))]
    [`(,f · ,g) `((,(δ f) · ,g) +  (,(δ g) · ,f))]))

; ` is unquote,   ,foo means to give value of variable foo

(δ '((cos + sin) · sin))








