#lang typed/racket

; Subtyping
; S <: T iff every value of type S can be used where a value of type T can.

; OO system, reusing implementation, and creating subtypes. Is complicated
; More of using Interface for expressing types rather than subtyping with inheritance

(: f1 : (U Real String) → (U Real String))
(: f2 : (U Real String) → Real)       ; more capable than f2, since accepts more args
(: f3 : Real → Real)
(: f4 : Real → (U Real String))

#; (define (f1 x) "hi")
(define (f1 x) 0)
(define (f2 x) 0)
(define (f3 x) 0)
(define (f4 x) 0)


; S <: T:  S guarantees more than T, S is more capable than T
;                        ->
; (U Real String) → Real <: (U Real String) → (U Real String)
; (U Real String) → Real <: Real → Real
;                                           <: Real → (U Real String)
; Real?(x) or String?(x) → Real?(f2(x))    ; a stronger constraint
; Real?(x) or String?(x) → Real?(f1(x)) or String?(f2(x))
; 
(add1 (f2 324))   ; safe
#; (add1 (f1 324))   ; not safe, regardless of implementation
(add1 (f2 324))
(add1 (f3 324))


; The type of a function is co-variant in the result types. (more safe to return more specific types)
; They vary in the same direction.
; The type of a function is contra-variant in the argument types. (more safe to accept more general types)




; Polymorphic Types

; Parametric Polymorphism, 

#| From 207 lectures on generics

    List list = new ArrayList();
    list.add("hello");
    String s = (String)list.get(0);

    List<String> list = new ArrayList<String>();
    list.add("hello");
    String s = list.get(0);    // without casting

 Type erasure:
    Type parameters go away after compile time.

 Casting:
    Making a claim on the type, a runtime value, will throw runtime exception
|#

; Listof is a type constructor: a function at compile-time
;    taking a type and producing a type

(require/typed typed/racket
               (first (∀ (α) (Listof α) → α))
               (rest (∀ (α) (Listof α) → (Listof α)))
               (map (∀ (α β) (α → β) (Listof α) → (Listof β)))
               (list (∀ (α) (α * → (Listof α)))))

(: a-list : (Listof Any))
(define a-list (list "hello"))
(first a-list)

; Type checking
; Just use symbols and call function at compile time

(inst list Boolean)  ; inst : instantiate
; - : (-> Boolean * (Listof Boolean))
; list is polymorphic,
; At compile, there are infinitely number of list functions for all types,
; can refer and use one at runtime, i.e. specifically for Boolean
; At runtime, the information is erased.

(define boolean-list (inst list Boolean))  ; compile time operation

(: ∘ : (∀ (α β γ) (β → γ) (α → β) → (α → γ)))
(define (∘ f g) (λ (x) (f (g x))))

; Unification: write down a set of equation for types
; solve the equations ... 
; f : D -> E
; g : B -> D
; x : B
; (g x) : D
; (f (g x)) : E
; (λ (x) (f (g x)))
;  B -> E
; (∘ f g) : (D -> E) (B -> D) (B -> E)

(define-type Stack (Listof Real))

; class Stack-Result<α>   (generic classes)
;    α result;
;    Stack stack;
(struct (α) Stack-Result ((result : α) (stack : Stack)) #:transparent)

(: pop : (Stack → (Stack-Result Real)))
(define (pop stack)
  (match stack [(list* top below) (Stack-Result top below)]))

(pop '(324 263 207))


; then,
; Takes 2 stack-affecting operation, then yielding a stack-affecting operation
; (: >>> : (∀ (α β)
;             (Stack → (Stack-Result α))
;             (Stack → (Stack-Result β))
;             → (Stack → (Stack-Result β))))
; (define (((>>> op-1) op-2) a-stack)
;   (match (op-1 a-stack) [(Stack-Result _ new-stack) (op-2 new-stack)]))
; 
; (((>>> pop) pop) '(3 2 4))


