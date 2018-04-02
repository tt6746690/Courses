#lang racket

(module+ test (require rackunit))

#;(type-of '(let [f (λ (x) x)]
              (let [b #t]
                (let [seq (λ (_) (λ (x) x))]
                  ((seq
                    (set! f (λ (x) 3)))
                   (set! b (f b)))))))


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
 |  
 |  Most general type of `f' : (∀ (α0) ((var α0) → (var α0))))
 |                       `b' : (∀ () bool)
 |  (α0 → int) ≡ (α0 → α0)  ⇒  (α0 ≡ α0) and (int ≡ α0)  ⇒  (α0 ≡ int)

 |  Idea: f is polymorphic, mutation to `f' 
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
 | Since ∀ can only appear in the outermost position in our type system,
 | we cannot give call/ec this type! We can get around this if we give
 | `k' its own special type (cont int) (or (cont (var α))) and add a special
 | `jump' function for invoking continuations:
 |
 |   > (call/ec (λ (k) (if ((jump k) 3) 0 (+ 1 ((jump k) 4)))))
 |
 | Devise most general types for call/ec and jump and add them to the builtin table.
 | Then, update subst, free-vars, and solve to handle the new type (just as you did with list).
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
    #;[call/ec ...]
    #;[jump    ...]))


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
; (instantiate '(∀ (α1 α2) (α1 → int)))   ; '(α1 → int)

; generalize : mono-type (list-of (symbol × poly-type)) (list-of constraint) → poly-type
(define (generalize A Γ G)
  ; We can only generalize over variables not in Γ.
  ; However, we don't know which variables are in Γ until we've solved all constraints.
  (println "A ---- ")
  (println A)
  (println "Γ ---- ")
  (println Γ)
  (println "G ---- ")
  (println G)
  (define σ      (solve G))
  (println "σ ---- ")
  (println σ)
  (define A′     (subst-all σ A))
  (println "A′ ---- ")
  (println A′)
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
                         (values 'unit `((,A ≡ ,B) ,@G))]
    ; (A ≡ bool) (B ≡ C)  where  t1 : A  t2 : B  t3 : C
    [`(if ,t₁ ,t₂ ,t₃)   (define-values (A G₁) (infer t₁ Γ))
                         (define-values (B G₂) (infer t₂ Γ))
                         (define-values (C G₃) (infer t₃ Γ))
                         (values C `((,A ≡ bool) (,B ≡ ,C) ,@G₁ ,@G₂ ,@G₃))]
    [`(let [,x ,t1] ,t2) (define-values (A G1) (infer t1 Γ))
                         ; (println `("====> " ,x " has type " ,(generalize A Γ G1)))
                         (define-values (B G2) (infer t2 `((,x ,(generalize A Γ G1)) ,@Γ)))
                         (values B `(,@G1 ,@G2))]))

#; '((b (∀ () bool))
     (+ (∀ () (int → (int → int))))
     (* (∀ () (int → (int → int))))
     (< (∀ () (int → (int → bool))))
     (∧ (∀ () (bool → (bool → bool))))
     (∨ (∀ () (bool → (bool → bool))))
     (¬ (∀ () (bool → bool))))


; subst : symbol mono-type → ((mono-type → mono-type) or (constraint → constraint))
;
; Substitutes B for α in A. Does not support poly-types,
; as that would require capture-avoiding substitution.
(define (subst α B)
  (define (rec A)
    (match A
      [`(,C → ,D) `(,(rec C) → ,(rec D))]
      [`(,C ≡ ,D) `(,(rec C) ≡ ,(rec D))]
      [`(list ,α) `(list ,(rec α))]
      ; Handle list types here
      ; Handle cont types here
      [_          (if (equal? A `(var ,α)) B A)]))
  rec)


; subst, free-vars, and solve 

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
    [`(∀ (,αs ...) A) (remove* αs (free-vars A))]
    [`((cons ,A) ,L) `(,@(free-vars A) ,@(free-vars L))]
    ; Handle cont types here
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
                                       (println "((var ,α)  ≡ ,B) ----- ")
                                       (println `((,α ,(subst-all σ B)) ,@σ))
                                       `((,α ,(subst-all σ B)) ,@σ)]
    [`((,A        ≡ (var ,α))  ,G ...) (solve `(((var ,α) ≡ ,A) ,@G))]
    [`(((,A → ,B) ≡ (,C → ,D)) ,G ...) (solve `((,A ≡ ,C) (,B ≡ ,D) ,@G))]
    ; Handle list types here
    ; Handle cont types here
    [`((,A        ≡ ,B)        ,G ...) (error (~a "Cannot unify " A " and " B "."))]))

#;  (type-of '((cons 1) null))

#; (infer '((cons 1) null))
#; '(var α27)
#; '(((var α25)
      ≡ ((list (var α)) → (var α27)))
     (((var α24) → ((list (var α)) → (list (var α))))
      ≡ (int → (var α25))))

#; (((var α24) → ((list (var α)) → (list (var α))))
      ≡ (int → (var α25)))

; has G
#; '(((var α32) ≡ ((list (var α)) → (var α34)))
     (((var α31) → ((list (var α)) → (list (var α)))) ≡ (int → (var α32))))
     ; ((cons α) alist) ≡ (int → )
; has σ
#; '((α32 ((list (var α)) → (list (var α))))
     (α31 int) (α34 (list (var α))))

; built-in types
#; [null (∀ (α) (list (var α)))]
#; [cons (∀ (α) ((var α) → ((list (var α)) → (list (var α)))))]

(module+ test
  ; behavior of type-of
  (check-equal? (type-of '((λ (x) x) 1))
                '(∀ () int))
  (check-equal? (type-of '((+ 1) 1))
                '(∀ () int))
  (check-match (type-of 'null)
                `(∀ () (list (var ,α))))
  (check-match (type-of 'cons)
               `(∀ (,α<n>)
                   ((var ,α<n>) → ((list (var ,α<n>)) → (list (var ,α<n>))))))
  
  ; behavior of infer
  (check-match (local [(define-values (A G) (infer '(λ (x) x)))] `(,A ,G))
               `(((var ,α) → (var ,α)) ()))
  (check-match (local [(define-values (A G) (infer '((λ (x) x) 1)))] `(,A ,G))
               `((var ,y)
                 ((((var ,x) → (var ,x)) ≡ (int → (var ,y))))))
  (check-match (local [(define-values (A G) (infer 'cons))] `(,A ,G))
               `(((var ,α<n>) → ((list (var ,α<n>)) → (list (var ,α<n>))))
                 ()))
  
  ; (infer '(λ (x) 1))          → '((var α0) → int)
  ; (infer '(λ (x) (λ (y) #t))) → '((var α1) → ((var α2) → bool))
  ; (infer '((λ (x) x) 1))
  ; '(var α4)
  ; '((((var α3) → (var α3)) ≡ (int → (var α4))))
  )

