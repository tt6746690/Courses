#lang racket #| CSC 324 - 2017 Fall - Assignment 1 - Part 2 / 2 |#

#| Due Saturday October 28th at Noon.
   May be done with one other student in the course. |#

#| ★ Implement ‘Eva’, a memory model based interpreter for an augmented Lambda Calculus.

 The syntactic language of Terms is the same as in Part 1, except for (add1 <term>).

 A run-time value is one of:
   • a number
   • a closure: (closure: (<λ-address> (<id>) <body>) <environment>)
     - a list containing the symbol closure:, a λ term with the symbol λ replaced by a unique symbol,
        and an environment
   • an address of a closure

 An environment is a list of two-element lists: ((<id> <value>) ...).
   Earlier <id>s are more local than later ones, i.e. they shadow later ones.
   The <value>s are
        * numbers or
        * addresses (<closure-address>)

 The model maintains a table associating addresses to closures.
 
 The semantics of function call is still eager by-value, with arguments passed by augmenting
  an environment.

   (<f> <a>)
     1. Evaluate <f> in the current environment, assume it produces an address of a closure c.
     2. Evaluate <a> in the current environment.
     3. Produce the value of the body of the λ term in c, evaluated in the environment of c
         augmented with a local variable associating the λ term's parameter with the value of <a>.

   (λ (<id>) <body>)
     1. Create a *new closure* with a new address, containing this λ term and current environment.
     2. Add the association of the address with the closure to the table of closures.
     3. Produce the address of the closure.

   <id>
     1. Produce the most local value of <id> from the current environment.

   <number>
     1. Produce the number. |#

(provide Eva)

#| Support library. |#

; table:   [ binding ... ]
; key: <closure address>
; binding: (<closure address>, <closure>)
; returns: <closure>
(define (lookup key table)
  (second (first
           (filter (λ (binding) (equal? (first binding) key))
                   table))))

(define (addresses)
  (define n -1)
  (λ () (local-require (only-in racket/syntax format-symbol))
    (set! n (add1 n))
    (format-symbol "λ~a" n)))

(module+ test
  (require rackunit)
  
  (check-equal? (lookup 'a '((b c) (a d) (a e))) 'd)

  (define generate-address (addresses))
  (check-equal? (generate-address) 'λ0)
  (check-equal? (generate-address) 'λ1))


#| Design and testing for Eva. |#

(module+ test
  
  ; An example, that is much too complicated for the first test in test-driven development.
  #;(check-equal? (Eva '(
                         (λ (x)
                           (λ (y) x))
                         (λ (y) 324)
                         )
                       )
                  '(λ2 ; Result value.
                    ; Table associating addresses to closures.
                    ((λ2 (closure: (λ (y) x) ((x λ1))))
                     (λ1 (closure: (λ (y) 324) ()))
                     (λ0 (closure: (λ (x) (λ (y) x)) ())))))

  ; term is a number
  (check-equal? (Eva 1)
                `(1 ()))
  ; term is an id,
  ; Note this test is invalid, because we assume id in current env
  #; (check-equal? (local [(define x 2)]
                     (Eva x))
                   `(2
                     ((x 2))))

  ; pick the most local env
  (check-equal? (Eva '(λ (y) x) '((x λ1)))
                 '(λ0
                   ((λ0 (closure: (λ (y) x) ((x λ1)))))))

  ; match for (λ (arg) body)
  (check-equal? (Eva '(λ (x) x))
                  '(λ0
                    ((λ0 (closure: (λ (x) x) ())))))

  ; exercise 5.2 example
  (check-equal? (Eva '((λ (One)
                         ((λ (Add1)
                            (Add1 (Add1 One)))
                          (λ (f)
                            (λ (g)
                              (λ (h) (g ((f g) h)))))))
                       (λ (h) h)))
                '(λ5
                  ((λ5 (closure: (λ (g) (λ (h) (g ((f g) h))))
                                 ((f λ4) (Add1 λ3) (One λ1))))
                   (λ4 (closure: (λ (g) (λ (h) (g ((f g) h))))
                                 ((f λ1) (Add1 λ3) (One λ1))))
                   (λ3 (closure: (λ (f) (λ (g) (λ (h) (g ((f g) h))))) ((One λ1))))
                   (λ2 (closure: (λ (Add1) (Add1 (Add1 One))) ((One λ1))))
                   (λ1 (closure: (λ (h) h) ()))
                   (λ0
                    (closure:
                     (λ (One)
                       ((λ (Add1) (Add1 (Add1 One))) (λ (f) (λ (g) (λ (h) (g ((f g) h)))))))
                     ())))))
  
  )

#| Eva takes a term and produces:
     1. Its value, which is an address or number.
     2. The table associating addresses to closures. |#

(define (Eva term [env '()]) ; default env for testing 

  (define generate-address (addresses))
  
  (define closures '())
  
  (define (eva term env)
    (define (make-closure λ-term env)
      `(closure: ,λ-term ,env))
    (define (closure-λ-term key)
      (second (lookup key closures)))
    (define (closure-λ-body key)
      (third (closure-λ-term key)))
    (define (closure-λ-arg key)
      (first (second (closure-λ-term key))))
    ; given a list of envs ((id′, value) ... ),
    ; return a filtered list where id′ matches supplied id
    (define (filter-env-by-id env id)
      (filter (λ (id-value) (equal? id (first id-value))) env))
    (match term
      [`(,f-term ,a-term) (local [(define f-closure (eva f-term env))
                                  (define a-closure (eva a-term env))
                                  (define f-body (closure-λ-body f-closure))
                                  (define f-arg (closure-λ-arg f-closure))
                                  (define new-env (list* (list f-arg a-closure) env))]
                            (eva f-body new-env))]
      [`(λ (,id) ,body) (local [(define closure-address (generate-address))]
                          (set! closures (list* (list closure-address (make-closure term env))
                                                closures))
                          closure-address)]
      [_ (cond [((λ (bool) (not (empty? bool)))
                 (filter-env-by-id env term))
                (second (first (filter-env-by-id env term)))]
               [else term])]))
  
  (list (eva term env) ; The value of term, in an empty initial environment.
        ; The table associating addresses to closures.
        closures))


#| ★ Implement ‘Eva’, a memory model based interpreter for an augmented Lambda Calculus.

 The syntactic language of Terms is the same as in Part 1, except for (add1 <term>).

 A run-time value is one of:
   • a number
   • a closure: (closure: (<λ-address> (<id>) <body>) <environment>)
     - a list containing the symbol closure:, a λ term with the symbol λ replaced by a unique symbol,
        and an environment
   • an address of a closure

 An environment is a list of two-element lists: ((<id> <value>) ...).
   Earlier <id>s are more local than later ones, i.e. they shadow later ones.
   The <value>s are
        * numbers or
        * addresses (<closure-address>)

 The model maintains a table associating addresses to closures.
 
 The semantics of function call is still eager by-value, with arguments passed by augmenting
  an environment.

   (<f> <a>)
     1. Evaluate <f> in the current environment, assume it produces an address of a closure c.
     2. Evaluate <a> in the current environment.
     3. Produce the value of the body of the λ term in c,
         evaluated in the environment of c
         augmented with __a local variable associating the λ term's parameter with the value of <a>__

   (λ (<id>) <body>)
     1. Create a *new closure* with a new address, containing this λ term and current environment.
     2. Add the association of the address with the closure to the table of closures.
     3. Produce the address of the closure.

   <id>
     1. Produce the most local value of <id> from the current environment.

   <number>
     1. Produce the number. |#