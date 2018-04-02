#lang racket

(require "basic-Mar-22.rkt")

#| Parametric Polymorphism/Generics
 | ================================
 |
 | In our simple type system, what type should we put in the _?
 |
 | > (λ (x : _) x)
 |
 | If we pick `int', it only works with integers.
 | If we pick `bool', it only works with booleans.
 |
 | We would like to parameterize functions by types so that they can work on any type.
 |
 | Our new language consists of the following terms:
 |
 | • x             (variables)
 | • n             (integers)
 | • b             (booleans)
 | • (λ (x : A) t)
 | • (app t₁ t₂)
 | • (set! x t)
 | • (if t₁ t₂ t₃)
 | • (Λ (α) t)     **NEW**  (type abstraction)
 | • (spec t A)    **NEW**  (type application)
 |
 | and the following types:
 |
 | • int
 | • bool
 | • unit
 | • (A → B)
 | • (var α)   **NEW**  (type variables)
 | • (∀ (α) A) **NEW**  (universal types)
 |
 | At some point we need to specialize ∀-types for specific values of `α'. For example, the following
 |
 | > (Λ (α) (λ (x : α) x))
 |
 | should have type `(∀ (α) (α → α))', while
 |
 | > (spec (Λ (α) (λ (x : α) x))
 |         int)
 |
 | should have type `(int → int)'. We must obtain `(int → int)' from `(α → α)' by substituting `int'
 | for `α'. Substitution becomes complicated when variable shadowing is involved. For example, what
 | happens if we substitute the type `(α → int)' for `β' in the expression `(∀ (α) (β → α))'? Naive
 | subtitution gives us
 |
 | > `(∀ (α) ((α → int) → α))'
 |
 | but the correct answer should be
 |
 | > `(∀ (γ) ((α → int) → γ))'
 |
 | We are going to get handle this by renaming variables as we encounter them.
 |
 |#

; rename a type (A) s.t. α is replaced with β
(define (rename A α β)
  (define (rename′ A) (rename A α β))
  (match A
    [`(,B → ,C)   `(,(rename′ B) → ,(rename′ C))]
    [`(∀ (,γ) ,B) `(∀ (,γ) ,(if (equal? α γ) B (rename′ B)))]
    [_            (if (equal? A α) β A)]))

; (rename '(int → bool) 'int 'bool) ; '(bool → bool)

; rename a type (A) s.t. α is replaced with another type B
; same behavior with rename, with 1 difference with how ∀ is handled
(define (subst A α B)
  (define (subst′ A) (subst A α B))
  (match A
    [`(,C → ,D)   `(,(subst′ C) → ,(subst′ D))]
    [`(∀ (,β) ,C) (define γ (gensym))
                  (define C′ (rename C β γ))   ; rename arg type β → γ in body C
                  `(∀ (,γ) ,(subst′ C′))]      ; ERROR add dot before subst
    [_            (if (equal? A α) B A)]))

; (subst '(int → bool) 'int '(int → int))  ; '((int → int) → bool)
; (subst '(∀ (α) (β → α)) 'β '(int → int)) ; '(∀ (γ) ((int → int) → γ))

(define (type? A [Δ builtin-types])
  (match A
    [`(,B → ,C)   (and (type? B) (type? C))]
    [`(∀ (,α) ,B) (type? B `(,α . ,Δ))]      ; add α to Δ
    [_            (member A Δ)]))

(define (type-of t [Δ builtin-types] [Γ builtin-terms])
  (define (type?′   A) (type?   A Δ))
  (define (type-of′ t) (type-of t Δ Γ))
  (type-of '(Λ (α) (λ (x : (var α)) x)))(match t
    [(? symbol? x)     (lookup x Γ)]
    [(? boolean?)      'bool]
    [(? integer?)      'int]
    [`(λ (,x : ,A) ,t) (assert (and (symbol? x) (type?′ A)))
                       `(,A → ,(type-of t Δ (cons (list x A) Γ)))]  ; add (x A) to Γ in case of λ
    [`(app ,t₁ ,t₂)    (match (type-of′ t₁)
                         [`(,A → ,B) (assert (equal? A (type-of′ t₂)))
                                     B]
                         [_          (assert #f)])]
    [`(set! ,x ,t)     (assert (and (symbol? x)
                                    (equal? (lookup x) (type-of′ t))))
                       'unit]
    [`(if ,t₁ ,t₂ ,t₃) (define A (type-of′ t₃))
                       (assert (and (equal? (type-of′ t₁) 'bool)
                                    (equal? (type-of′ t₂) A)))
                       A]
    [`(Λ (,α) ,t)      `(∀ (,α) ,(type-of t (cons α Δ) Γ))]
    [`(spec ,t ,A)     (match (type-of′ t)
                         [`(∀ (,α) ,B) (assert (type?′ A))
                                       (subst B α A)]
                         [_            (assert #f)])]))
