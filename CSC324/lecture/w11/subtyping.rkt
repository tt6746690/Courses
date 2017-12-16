#lang typed/racket #| Static Typing: Subtyping |#

#| This is an expanded summary of the union of the discussions of these topics
    between the two sections of the course.

 References to OO are to:
   • leverage your intuition and discussions from previous courses
   • allow you to apply this in situations that might look superficially different

 References to logic are for similar reasons.

 Testable exam material is:
   • determining the subtype relationship, if any, between particular types
   • the concrete consequences, illustrated by whether concrete code type checks
      and/or is generally “safe” |#


#| “S <: T” is the notation for: “S is a subtype of T”.

 Recall the Liskov substitution principle. This was almost certainly mentioned, by name,
  in the CSC148 and/or CSC207 you took, in the context of Object Orientation, subclassing,
  and polymorphism.

   If ϕ is a unary predicate then:
     If ϕ is true for all values of type T, then ϕ is true for all values of type S.
     I.e., (∀ t of type T, ϕ(t)) → (∀ s of type S, ϕ(s)).

 In the set-of-values view of types:  S <: T  ↔  S ⊂ T.

 In the predicate view, let Φ(T) mean: ∀t of type T, ϕ(t), i.e. “ϕ is true for values of type T”.
 Then  S <: T  ↔  Φ(T) → Φ(S).

 Some basic properties:
   T <: T - reflexive
   ( S <: T  ∧  T <: U ) → ( S <: U ) - transitive |#


#| Simple values. |#

; Real   <: (U Real String)
; String <: (U Real String)
; Neither of Real and String is a subtype of the other.
; If we have reasoned about something knowing only that it's a Real or String, then the knowledge
;  applies to any Real, and any String.

(: r : Real)
(: s : String)
(: r-or-s-1 : (U Real String))
(: r-or-s-2 : (U Real String))
(: r-or-s-3 : (U Real String))

(define r 324)
#;(define r "bee") ; ⊥
#;(define s 324)   ; ⊥
(define s "bee")
(define r-or-s-1 r)
(define r-or-s-2 s)
(define r-or-s-3 (if (zero? (random 2)) r s))

; Unsafe, based on the declarations.
#;(set! r r-or-s-1)
#;(set! r r-or-s-2)
#;(set! r r-or-s-3)

; Safe.
(set! r-or-s-3 r)


#| Function values. |#

(: fuu : (U Real String) → (U Real String)); 2
(: fru :    Real         → (U Real String)); 4
(: fur : (U Real String) →    Real)        ; 1 
(: frr :    Real         →    Real)        ; 3

(define (fuu a) 0)
(define (fru a) 0)
(define (fur a) 0)
(define (frr a) 0)

; Which function is safe to use in the most situations?

; fur handles the most,
(fur 324)
(fur "bee")
; and guarantees the most:
(add1 (fur 324))

; Unsafe: since might return String 
#;(add1 (fuu 324))
#;(add1 (fru 324))
; Those are okay at run time based on the current implementations of fuu and fru.
; So Exercise: change their definitions, without changing their type declarations,
;  so that their definitions still type check, but those expressions would error
;  at run time if allowed.

; Unsafe: since not accepting String
#;(fru "bee")
#;(frr "bee")
; Those are okay at run time based on the current implementations of fru and frr.
; So Exercise: change their definitions, without changing their type declarations,
;  so that their definitions still type check, but those expressions would error
;  at run time if allowed.

; ((U Real String) → Real)
;   <: (Real → Real), ((U Real String) → (U Real String))
;      <: (Real → (U Real String))
;
; f ∈ ((U Real String) → Real)  →  f ∈ (Real → Real) ∧ f ∈ ((U Real String) → (U Real String))
; f ∈ (Real → Real) ∨ f ∈ ((U Real String) → (U Real String))  →  f ∈ ((U Real String) → Real)
; Notice the ∧ in the first line, versus the ∨ in the second line.

; The use of “→” for both function types and implications is not a coincidence.
;
; In the set view, for function types S and T, S is some set of (λ (a) body-expr ...)s,
;  and S <: T means each one of those is in T as well.
; E.g., the set of valid bodies for fur is a subset of the valid bodies for frr.
; That feels nice and “operational”.
;
; The predicate view is very valuable here even if you're less comfortable with logic and implication.
; Recall:
;   If A → B, then  (C → A) → (C → B) [think about proving (C → B) from (A → B) ∧ (C → A)],
;              but  (B → C) → (A → C).
;   In particular, since (A ∧ B) → A, and (A ∧ B) → B, and A → (A ∨ B), and B → (A ∨ B):
;                   (C → (A ∧ B)) → (C → A) [if I know C then I know A and B, so if I know C I know A]
;                   (C → (A ∧ B)) → (C → B)
;                   (C → A) → (C → (A ∨ B))
;                   (C → B) → (C → (A ∨ B))
;                   ((A ∨ B) → C) → (A → C)
;                   ((A ∨ B) → C) → (B → C)
;                   (A → C) → ((A ∧ B) → C)
;                   (B → C) → ((A ∧ B) → C)
;
; I didn't actually recall those: when they come up they rarely “look” like the above anyway.
; Some courses might ask you about them explicitly, but the “real world” rarely does.
; I derive them each time, which develops and memorizes the meaning. In particular, I wrote them
;  out *after* writing out the material below. I don't even know whether I've read an explicit
;  treatment of most of the material below, because if I did it would have been stored as
;  “working out the consequences of assuming the pre-condition and knowing the post-condition
;   is exactly what it sounds like: the consequence of an implication”.

; Let's view Real and String as predicates. Then, e.g., ∀ a, [Real(a) ∨ String(a)] → Real(fur a).
;
; Consider predicates defined by:
;   pur(f) ≡ ∀ a, Real(a) ∨ String(a) → Real(f(a))
;   prr(f) ≡ ∀ a, Real(a)             → Real(f(a))
;   puu(f) ≡ ∀ a, Real(a) ∨ String(a) → Real(f(a)) ∨ String(f(a))
;   pru(f) ≡ ∀ a, Real(a)             → Real(f(a)) ∨ String(f(a))
; Those are predicate versions of the function types, expressed in terms of the component types.
;
; Then:
;   ∀ f, pur(f) → prr(f) ∧ puu(f)
;   ∀ f, [prr(f) ∨ puu(f)] → pru(f)
;
; It's also not a coincidence that the summary had to change ‘∨’ to ‘∧’ when moving it from the
;  right to the left side of the implication. Conjunction in a conclusion strengthens an
;  implication, but weakens it 

; Implications, and function types, are “Covariant” in their conclusions / result types:
;   If S <: T, then (X → S) <: (X → T).
;   If A →  B, then (C → A) →  (C → B).

; But they are “Contravariant” in their hypotheses / domain types:
;   If S <: T, then (X → S) :> (X → T).
;   If A →  B, then (C → A) ←  (C → B).
; Notice the direction of subtyping/implication flipped.

; Covariance and Contravariance are common and powerful concepts, frequent in typing, but also
;  in other aspects of programming, in logic, and any time things can “vary” in some way:
;   “When we vary this aspect, does this other aspect vary in the same way, opposite way, or neither?”
;
; Many statically typed OO languages allow a subclass to override a method
;  and change the return type to a subtype.
; Some also allow an overriding method to change the argument type(s) to supertype(s).
;
; A generic/template is neither covariant nor contravariant in one of its types, unless that type
;  variable is used only for parameter types, or only for result types.
; Instance variables are a special case: they can be typed by thinking of them via pairs of
;  setter and getter, or just getter or just setter if they are read-only or write-only to the public.


#| Mutable values.

 Is Square <: Rectangle, or is Rectangle <: Square, or neither?
 Search the web for that: it's a frequent source of confusion and unproductive debate.

 We'll answer it, but first for the simplest case: a container that contains a single
  value, where the container can be changed to hold a different value.

 Is a box of tigers a box of kittens? Is a box of kittens a box of tigers?
 (Boxof Tiger)  <: (Boxof Kitten)?
 (Boxof Kitten) <: (Boxof Tiger)? |#

; Re-import with slightly simpler types for this discussion:
(require/typed racket
               ; Constructor:
               (box      (All (a) (a -> (Boxof a))))
               ; Getter:
               (unbox    (All (a) ((Boxof a) -> a)))
               ; Setter:
               (set-box! (All (a) ((Boxof a) a -> Void))))

(: bu : (Boxof (U Real String)))
(: br : (Boxof Real))
(define bu (box 324))
(define br (box 324))

; We can explicitly select a particular version of the parametric functions with ‘inst’.
; In typing, people view, e.g., unbox as a compile time function that can be called [at compile time],
;  where the ‘a’ in ‘(All (a) _)’ designates is the “header”, producing a particular “concretely”
;  typed version of the function.
#;(unbox (All (a) ((Boxof a) -> a)))
(ann (inst unbox Real) ((Boxof Real) → Real))
(: unbox-real-1 : ((Boxof Real) → Real))
(: unbox-real-2 : ((Boxof Real) → Real))
(define unbox-real-1 (inst unbox Real))
(define unbox-real-2 unbox) ; A bit of type inference: declaration above forces ‘a’.

; In Java, imagine class Boxof<T>.
;   Boxof<double> br = new Boxof<double>(324); // Particular version of constructor.
;   double r = br.unbox() ; // At compile time, considers the method to be: double unbox()
; There is also an implicit ‘this’ parameter, explicit as ‘self’ in python, which means unbox
;   is more explicitly like:
;   static T unbox<T>(Boxof<T> b)
;  and the particular version unbox<double> is like:
;   static double unbox(Boxof<double> b)

; Look now at set-box!:
#;(set-box! (All (a) ((Boxof a) a -> Void)))
(ann (inst set-box! Real) ((Boxof Real) Real -> Void))

; The result type for unbox, recalling the covariance discussion above, needs T <: Real
;  for (Boxof T) <: (Boxof Real).
; The second argument's type in set-box!, recalling the discussion above, needs Real <: T
;  for (Boxof T) <: (Boxof Real).
; So there's no subtyping relationship between different types of boxes.


; As with so much of computing, reasoning about the lambda calculus [functions and calls]
;  covers a lot of situations.
;
; But many languages obscure the commonalities:
;   • inventing new syntaxes for familiar constructs
;   • implementing constructs that combine the common fundamental ones, without explaining
;      the constructs in terms of the fundamental ones
;
; So it's worth being able to think well about the “mutable container” type, and easily discuss
;  it with untrained programmers in terms they're used to, and have examples handy for them.

#;(set! br bu)
#;(set! bu br)

(add1 (unbox br))
#;(add1 (unbox bu)) ; Safe, for now.
(set-box! bu "cat")
#;(set-box! br "cat")
#;(add1 (unbox bu)) ; Not safe in general, based only on declared type.
                    ; an argument that follows contravariance doenst always work

(: f : (Boxof (U Real String)) → Void)
(define (f b) (set-box! b "string"))
#;(f br) ; problem! an argument that doesnt follow contravariacne doesnt work either
(: g : (Boxof Real) → Void)
(define (g b)
  (add1 (unbox b))
  (void))
#;(g bu) ; problem! 

; If you have a box that can only hold a kitten, and you give it to someone who expects a box that
;  can hold a tiger, they'll be upset when they try to put the tiger in the inadequate box.
; If you give someone a box than can hold a tiger, and it contains a kitten, when they give it back
;  you might be upset.
; Mutation can really come back to bite you!

; As for rectangles and squares:
;   A mutable square is not a mutable rectangle: the sides can't be mutated independently.
;   A rectangle is not a square: you can't assume the sides are the same.
; So for mutable shapes, neither is a subtype of the other.
; For immutable shapes, every square is a rectangle: Immutable-Square <: Immutable-Rectangle.
; This is derivable from the method types, thinking of them as just function types.
; You can now safely ignore the endless online discussions about it.

; The List type in typed racket is immutable.
; The parametric type (Listof T) is covariant in the type T.
; So (Listof Real) <: (Listof (U Real String)), since Real <: (U Real String).
(: lr : (Listof Real))
(define lr (list 1 2 3))
(: lu : (Listof (U Real String)))
(define lu lr)
(or (empty? lr) (add1 (first lr)))
#;(or (empty? lr) (add1 (first lu))) ; Safe here, but what are we worried lu could have been?
#;(set! lr lu) ; And then this could lead to unsafety.
