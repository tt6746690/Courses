#lang racket

(provide builtin-terms builtin-types assert lookup)

#| Basic Type Checking
 | ===================
 |
 | The simplest type checker consists of a single function
 |
 | > type-of : term → type (or error)
 |
 | Our language consists of the following terms:
 |
 | • x             (variables)
 | • n             (integers)
 | • b             (booleans)
 | • (λ (x : A) t)
 | • (app t₁ t₂)
 | • (set! x t)
 | • (if t₁ t₂ t₃)
 |
 | and the following types:
 |
 | • int
 | • bool
 | • unit
 | • (A → B)
 |
 |#

(define builtin-terms
  '([+ (int  → (int  → int))]
    [* (int  → (int  → int))]
    [< (int  → (int  → bool))]
    [∧ (bool → (bool → bool))]
    [∨ (bool → (bool → bool))]))

(define builtin-types '(int bool unit))

(define (assert b)
  (unless b
    (error "Type checking error!")))

; return type of term with name x in context Γ
(define (lookup x Γ)
  (define p (assoc x Γ))  ; look up first element in Γ of form (x type)
  (assert p)
  (second p))           

; A is a type if either its a function type or a built-in type (int bool unit)
(define (type? A)
  (match A
    [`(,A → ,B) (and (type? A) (type? B))]
    [_          (member A builtin-types)]))

(define (type-of t [Γ builtin-terms])
  (define (type-of′ t) (type-of t Γ))
  (match t
    [(? symbol? x)     (lookup x Γ)]
    [(? boolean?)      'bool]
    [(? integer?)      'int]
    [`(λ (,x : ,A) ,t) (assert (and (symbol? x) (type? A)))
                       `(,A → ,(type-of t `((,x ,A) . ,Γ)))]   ; add (x A) to env
    [`(app ,t₁ ,t₂)    (match (type-of′ t₁)
                         [`(,A → ,B) (assert (equal? A (type-of′ t₂)))
                                     B]
                         [_          (assert #f)])]
    [`(set! ,x ,t)     (assert (and (symbol? x)
                                    (equal? (lookup x Γ) (type-of′ t))))
                       'unit]
    [`(if ,t₁ ,t₂ ,t₃) (define A (type-of′ t₃))
                       (assert (and (equal? (type-of′ t₁) 'bool)
                                    (equal? (type-of′ t₂) A)))
                       A]))




#| In the literature, type checking rules are written formally using combinations of the following:
 |
 |    Hypothesis₁
 |    Hypothesis₂
 |    ...
 | • ─────────────
 |    Conclusion
 |
 | • Γ ⊢ t : A
 |
 | The first means "if the hypothesis are true, then so is the conclusion".
 | The second means "t has type A in environment Γ"
 |  (where Γ is usually associates variables with types).
 |
 | The type checking rules for the above language would be written formally as:
 |
 |    x : A ∈ Γ
 | • ───────────
 |    Γ ⊢ x : A
 |
 |
 |    n ∈ ℤ
 | • ─────────────
 |    Γ ⊢ n : int
 |
 |
 |    n ∈ {#t, #f}
 | • ──────────────
 |    Γ ⊢ b : bool
 |
 |
 |    x : A, Γ ⊢ t : B
 | • ─────────────────────────────
 |    Γ ⊢ (λ (x : A) t) : (A → B)
 |
 |
 |    Γ ⊢ t₁ : (A → B)
 |    Γ ⊢ t₂ : A
 | • ─────────────────────
 |    Γ ⊢ (app t₁ t₂) : B
 |
 |
 |    x : A ∈ Γ
 |    Γ ⊢ t : A
 | • ───────────────────────
 |    Γ ⊢ (set! x t) : bool
 |
 |
 |    Γ ⊢ t₁ : bool
 |    Γ ⊢ t₂ : A
 |    Γ ⊢ t₃ : A
 | • ───────────────────────
 |    Γ ⊢ (if t₁ t₂ t₃) : A
 |
 |#
