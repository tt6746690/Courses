#lang racket

(module+ test (require rackunit))


#| Assignment 3 (replacement)
 | ============
 |#

#| Question 1
 | ~~~~~~~~~~
 |
 | The following implements type inference with let-generalization, as described in class.
 | As mentioned in class, it has a problem. The following passes the type checker:
 |
 |   > (type-of '(let [f (λ (x) x)]
 |                 (let [b #t]
 |                   (let [seq (λ (_) (λ (x) x))]
 |                     ((seq
 |                        (set! f (λ (x) 3)))
 |                        (set! b (f b)))))))
 |   '(∀ () unit)
 |
 | However, at runtime this would result in `b' being assigned the value 3, and `b' is a bool!
 |
 | Describe how you would prevent this problem, and implement your solution.
 |                                                  -----------------------
 | Your solution should at least:
 |
 | • Disallow mutation of `f'
 | • Allow mutation of `b'
 |
 | HINT: Use type-of to determine the most general types of `f' and `b'. How do they differ?
 |
 | Type your description in the space below:
 |#

#|

  Given (set! v e), we evaluate (type-of v) and (type-of e) in example's context

       (type-of v)                       (type-of e)
  f :  '(∀ (α0) ((var α0) → (var α0)))   '(∀ (α0) ((var α0) → int))
  b :  '(∀ () bool)                      '(∀ () bool)

  Idea is because we use let-generalization.
  The type-checker assigns f initialized in let as ∀α. (α → α)
  The type-checker does not complain (set! b (f b)) after f is mutated to ∀α. (α → int),
    because it still regard f's type as the generalized ∀α. (α → α)
  One way to prevent this is to disallow mutation that assigns f to a non alpha equivalent type
    such that application of f elsewhere is valid
|#


#| Question 2
 | ~~~~~~~~~~
 |
 | We would like to support lists of a given type. Let us denote them as (list A).
 |
 | • (list int)          is the type of lists of ints.
 | • (list (int → bool)) is the type of lists of functions from ints to bools.
 | • (list (var α))      is the type of lists of αs.
 |
 | First, add `null' and `cons' to the list of builtins with the appropriate, most
 | general type. To start you off, null should have type (∀ (α) (list (var α))).
 |
 | Second, update subst, free-vars, and solve to handle the new type.
 |
 | HINT: Take a look at how function types are handled in subst, free-vars, and solve.
 |#


#| Question 3
 | ~~~~~~~~~~
 |
 | Recall call/ec:
 |
 |   > (call/ec (λ (k) (if (k 3) 0 (+ 1 (k 4)))))
 |
 | In this example, (k 3) has type bool and (k 4) has type int. We would like `k' to have
 | type (∀ (α) (int → (var α))), however to do this we would have to give call/ec the type
 |
 |   (((∀ (α) (int → (var α))) → int) → int)
 |
 | or the more general
 |
 |   (∀ (β) (((∀ (α) ((var β) → (var α))) → (var β)) → (var β)))
 |
 | [If you're getting lost in the parentheses, these can be more succinctly written as
 |
 |   ((∀α. int → α) → int) → int
 |
 |  and
 |
 |   ∀β. ((∀α. β → α) → β) → β]
 | 
 | i.e.   k                   : (∀α. β → α)                 ; (∀ (α) ((var β) → (var α)))
 |        (λ (k) ⋯)           : ((∀α. β → α) → β)           ; ((∀ (α) ((var β) → (var α))) → (var β))
 |        (call/ec (λ (k) ⋯)) : (∀β. ((∀α. β → α) → β) → β) ; (((∀ (α) ((var β) → (var α))) → (var β)) → (var β))
 |
 | Since ∀ can only appear in the outermost position in our type system,
 | we cannot give call/ec this type! We can get around this if we give
 | `k' its own special type (cont int) (or (cont (var α))) and add a special
 | `jump' function for invoking continuations:
 |
 |   > (call/ec (λ (k) (if ((jump k) 3) 0 (+ 1 ((jump k) 4)))))
 |
 | Devise most general types for call/ec and jump and add them to the builtin table.
 | Then, update subst, free-vars, and solve to handle the new type (just as you did with list).
 |
 |        α : type fitting situation where k is called, return type for (k arg)
 |        β : type of arg when evoking continuation k, also type of return value of call/ec
 | i.e.   k                     : (∀β. (cont β))
 |        jump                  : (∀αβ. (cont β) → (β → α))
 |        (λ (k) ⋯)             : (∀β. (cont β) → β)
 |        (call/ec (λ (k) ⋯))   : (∀β. ((cont β) → β) → β)
 |#


#| Data Structures
 | ~~~~~~~~~~~~~~~
 |
 | A poly-type is of the form
 |   ---------
 |
 | • (∀ (α ...) A)
 |
 | where
 |
 | - α ... are symbols
 | - A is a mono-type
 |
 | A mono-type is of one of the following forms
 |   ---------
 |
 | • (var α)
 | • (A → B)
 | • int
 | • bool
 | • unit
 |
 | where
 |
 | - α is a symbol
 | - A and B are mono-types
 |
 | A constraint is of the form
 |   ----------
 |
 | • (A ≡ B)
 |
 | where
 |
 | - A and B are mono-types
 |
 |#

; type-of : term (list-of (symbol × poly-type)) → poly-type
(define (type-of t [Γ builtins])
  (define-values (A G) (infer t Γ))
  (generalize A Γ G))

; builtins : (list-of (symbol × poly-type))
(define builtins
  '([+ (∀ () (int  → (int  → int)))]
    [* (∀ () (int  → (int  → int)))]
    [< (∀ () (int  → (int  → bool)))]
    [∧ (∀ () (bool → (bool → bool)))]    
    [∨ (∀ () (bool → (bool → bool)))]
    [¬ (∀ () (bool → bool))]
    ; Add list functions here
    [null (∀ (α) (list (var α)))]
    [cons (∀ (α) ((var α) → ((list (var α)) → (list (var α)))))]
    ; Add continuation functions here
    [call/ec (∀ (β) (((cont (var β)) → (var β)) → (var β)))]
    [jump (∀ (α β) ((cont (var β)) → ((var β) → (var α))))]))


; fresh : → mono-type  i.e. '(var α1) 
(define fresh
  (let [(c 0)]
    (λ ()
      (set! c (+ c 1))
      `(var ,(string->symbol (~a 'α (- c 1)))))))

; instantiate : poly-type → mono-type
(define (instantiate A)
  (match A
    [`(∀ (,αs ...) ,A) (define σ (map (λ (α) `(,α ,(fresh))) αs)) ; ((α1 (var α2)) (α2 (var α3)))
                       (subst-all σ A)]))

; generalize : mono-type (list-of (symbol × poly-type)) (list-of constraint) → poly-type
(define (generalize A Γ G)
  ; We can only generalize over variables not in Γ.
  ; However, we don't know which variables are in Γ until we've solved all constraints.
  (define σ      (solve G))
  (define A′     (subst-all σ A))
  ; subst-all does not support poly-types.
  ; Supporting them would require implementing capture-avoiding substitution.
  ; Instead we just instantiate every type in the environment.
  ; This will result in more free variables, but they would all be fresh and not appear in A anyways.
  (define Γ′     (map (compose (curry subst-all σ) instantiate second) Γ))
  (define A-vars (free-vars A′))
  (define Γ-vars (append-map free-vars Γ′))
  (define αs     (remove-duplicates (remove* Γ-vars A-vars)))
  `(∀ ,αs ,A′))

; lookup : symbol (list-of (symbol × poly-type)) → mono-type
(define (lookup x Γ)
  (define p (assoc x Γ))
  (unless p (error (~a "Variable " x " not in scope.")))
  (instantiate (second p)))


; check if 2 types are equivalent
(define (α≡ x y)
  (match-define `(∀ (,αsx ...) ,Ax) x)
  (match-define `(∀ (,αsy ...) ,Ay) y)
  (define σ (map (λ (x y) `(,x (var ,y))) αsx αsy))
  (and (equal? (length αsx) (length αsy))
       (equal? (subst-all σ Ax) Ay)))


; infer : term (list-of (symbol × poly-type)) → mono-type (list-of constraint)
;
; infer returns:
;
; • the (ungeneralized) type of t
; • a list of constraints that the variables in the type must satisfy
(define (infer t [Γ builtins])
  (match t
    [(? symbol?)         (values (lookup t Γ) '())]
    [(? boolean?)        (values 'bool        '())]
    [(? integer?)        (values 'int         '())]
    ; Assign a universal type to (x : A)
    ; Add x → (∀ () A) to Γ, Infer type of t
    [`(λ (,x) ,t)        (define A (fresh))
                         (define-values (B G) (infer t `((,x (∀ () ,A)) ,@Γ)))
                         (values `(,A → ,B) G)]
    ; Infer type for t1 : A and t2 B
    ; (A ≡ (B → C))  where  t1 : A  t2 : B
    [`(,t₁ ,t₂)          (define-values (A G₁) (infer t₁ Γ))
                         (define-values (B G₂) (infer t₂ Γ))
                         (define C (fresh))
                         (values C `((,A ≡ (,B → ,C)) ,@G₁ ,@G₂))]
    ; (A ≡ B)  where  x : A  t : B
    [`(set! ,x ,t)       (define A (lookup x Γ))
                         (define-values (B G) (infer t Γ))
                         (cond [(α≡ (type-of x Γ) (type-of t Γ))
                                (values 'unit `((,A ≡ ,B) ,@G))]
                               [else (error (~a "Cannot mutate "
                                                (type-of x Γ) " to " (type-of t Γ) "."))])
                         ]
    ; (A ≡ bool) (B ≡ C)  where  t1 : A  t2 : B  t3 : C
    [`(if ,t₁ ,t₂ ,t₃)   (define-values (A G₁) (infer t₁ Γ))
                         (define-values (B G₂) (infer t₂ Γ))
                         (define-values (C G₃) (infer t₃ Γ))
                         (values C `((,A ≡ bool) (,B ≡ ,C) ,@G₁ ,@G₂ ,@G₃))]
    [`(let [,x ,t1] ,t2) (define-values (A G1) (infer t1 Γ))
                         (define-values (B G2) (infer t2 `((,x ,(generalize A Γ G1)) ,@Γ)))
                         (values B `(,@G1 ,@G2))]))


; subst : symbol mono-type → ((mono-type → mono-type) or (constraint → constraint))
;
; Substitutes B for α in A. Does not support poly-types,
; as that would require capture-avoiding substitution.
(define (subst α B)
  (define (rec A)
    (match A
      [`(,C → ,D) `(,(rec C) → ,(rec D))]
      [`(,C ≡ ,D) `(,(rec C) ≡ ,(rec D))]
      ; Handle list types here
      [`(list ,α) `(list ,(rec α))]
      ; Handle cont types here
      [`(cont ,α) `(cont ,(rec α))]
      [_          (if (equal? A `(var ,α)) B A)]))
  rec)


; subst-all : (list-of (symbol × mono-type)) mono-type → mono-type
; substitute all occurrances of symbol in A to mono-type it maps to
(define (subst-all σ A)
  ((apply compose (map (curry apply subst) σ)) A))

; (compose proc ...) : return another λ s.t. proc evaluated in reverse order to args

; free-vars : type → (list-of symbol)
; return variables in a function that is not argument
(define (free-vars A)
  (match A
    [`(var ,α)        `(,α)]
    [`(,A → ,B)       `(,@(free-vars A) ,@(free-vars B))]
    [`(∀ (,αs ...) ,A) (remove* αs (free-vars A))]
    ; Handle list types here
    [`(list ,A)       (free-vars A)]
    ; Handle cont types here
    [`(cont ,A)       (free-vars A)]
    [_                '()]))




; solve : (list-of constraint) → (list-of (symbol × mono-type))
;
; Finds an assignment for each variable appearing in G such that:
;
; • symbols are only given one assignment
; • an assigned symbol is never mentioned on the right hand side of an assignment
(define (solve G)
  (match G
    ['() '()]
    [`((,A        ≡ ,B)        ,G ...) #:when (equal? A B)
                                       (solve G)]
    [`(((var ,α)  ≡ ,B)        ,G ...) #:when (not (member α (free-vars B)))
                                       ; Eliminates all occurrences of α in G before solving
                                       (define σ (solve (map (subst α B) G)))
                                       ; Eliminate all assigned variables in σ from B
                                       `((,α ,(subst-all σ B)) ,@σ)]
    [`((,A        ≡ (var ,α))  ,G ...) (solve `(((var ,α) ≡ ,A) ,@G))]
    [`(((,A → ,B) ≡ (,C → ,D)) ,G ...) (solve `((,A ≡ ,C) (,B ≡ ,D) ,@G))]
    ;   G ∪ { (list A) ≡ (list B) }   ⇒   G ∪ { A ≡ B }
    [`(((list ,A) ≡ (list ,B)) ,G ...) (solve `((,A ≡ ,B) ,@G))]
    ;   G ∪ { (cont A) ≡ (cont B) }   ⇒   G ∪ { A ≡ B }
    [`(((cont ,A) ≡ (cont ,B)) ,G ...) (solve `((,A ≡ ,B) ,@G))]
    [`((,A        ≡ ,B)        ,G ...) (error (~a "Cannot unify " A " and " B "."))]))


(module+ test
  ; equivalent type
  (check-equal? (α≡ '(∀ (α24) ((var α24) → (var α24)))
                    '(∀ (α33) ((var α33) → int)))
                #f)
  (check-equal? (α≡ '(∀ (α24) ((var α24) → (var α24)))
                    '(∀ (α33) ((var α33) → (var α33))))
                #t)
  (check-equal? (α≡ '(∀ () bool)
                    '(∀ () bool))
                #t)
  ; subst-all
  (check-equal? (subst-all '((α (var α171)) (β (var α172)))
                           '((cont (var β)) → ((var β) → (var α))))
                '((cont (var α172)) → ((var α172) → (var α171))))
  ; free-vars
  (check-equal? (free-vars '(var α))
                '(α))
  (check-equal? (free-vars '((var α) → (var β)))
                '(α β))
  (check-equal? (free-vars '(∀ (α) (var α)))
                '())
  (check-equal? (free-vars '(∀ (α) (var β)))
                '(β))
  (check-equal? (free-vars '(list (var α)))
                '(α))
  ; instantiate
  (check-equal? (instantiate '(∀ (α1 α2) (α1 → int)))
                '(α1 → int))
  ; behavior of type-of
  (check-equal? (type-of '((λ (x) x) 1))
                '(∀ () int))
  (check-equal? (type-of '((+ 1) 1))
                '(∀ () int))
  (check-match (type-of 'null)
               `(∀ (,α) (list (var ,α))))
  (check-match (type-of 'cons)
               `(∀ (,α<n>)
                   ((var ,α<n>) → ((list (var ,α<n>)) → (list (var ,α<n>))))))
  (check-equal? (type-of '((cons 1) null))
                '(∀ () (list int)))
  (check-equal? (type-of '((cons #t) ((cons #f) null)))
                '(∀ () (list bool)))
  (check-exn exn:fail? (λ () (type-of '((cons 1) ((cons #t) null)))))
  
  ; behavior of infer
  (check-match (local [(define-values (A G) (infer '(λ (x) x)))] `(,A ,G))
               `(((var ,α) → (var ,α)) ()))
  (check-match (local [(define-values (A G) (infer '((λ (x) x) 1)))] `(,A ,G))
               `((var ,y)
                 ((((var ,x) → (var ,x)) ≡ (int → (var ,y))))))
  (check-match (local [(define-values (A G) (infer 'cons))] `(,A ,G))
               `(((var ,α<n>) → ((list (var ,α<n>)) → (list (var ,α<n>))))
                 ()))

  ; q3
  (check-equal? (type-of '(call/ec (λ (k) 3)))
                '(∀ () int))
  (check-equal? (type-of '(call/ec (λ (k) ((jump k) 1))))
                '(∀ () int))
  (check-equal? (type-of '(call/ec (λ (k) (if ((jump k) 1)
                                              2
                                              ((jump k) 3)))))
                '(∀ () int))
  (check-exn exn:fail? (λ () (type-of '(call/ec (λ (k) (if ((jump k) 1)
                                                           2
                                                           ((jump k) #t)))))))

  ; q1
  (check-exn exn:fail? (λ () (type-of '(let [f (λ (x) x)]
                                         (let [b #t]
                                           (let [seq (λ (_) (λ (x) x))]
                                             ((seq
                                               (set! f (λ (x) 3)))
                                              (set! b (f b)))))))))
  (check-exn exn:fail? (λ () (type-of '(let [f (λ (x) (λ (y) x))]
                                         (let [b #t]
                                           (let [seq (λ (_) (λ (x) x))]
                                             ((seq
                                               (set! f (λ (x) (λ (y) 3))))
                                              (set! b ((f b) b)))))))))
  (check-equal? (type-of '(let [f (λ (x) x)]
                            (let [b #t]
                              (set! b (f b)))))
                '(∀ () unit))
  (check-equal? (type-of '(let [f (λ (x) #t)]
                            (let [b #t]
                              (let [seq (λ (_) (λ (x) x))]
                                ((seq
                                  (set! f (λ (x) b)))
                                 (set! b (f b)))))))
                '(∀ () unit))
  
  )
