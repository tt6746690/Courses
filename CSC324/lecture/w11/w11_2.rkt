#lang typed/racket #| Static Typing |#

#| Software Verification

   Without "running": check some properties of a program.
     Halting Problem and others, mean: impossible in general,
     and there are also tractibility problem.

   Static: at "compile" time, i.e without "running".
   Typing: some language of claims, completely decidable in "reasonable" time,
    separately compilable and typeable.
   
|#


; One common static check: syntax
; bash doesn't check
; script:
;    echo 'hi'
;    for            # raise error

; Java: variable declared and obviously initialized
;    int a;
;    if (true) { a = 1; }   // not allowed

; C: variable declared
; Python: neither checked
(displayln "hi")
#; x  ; unbuond variable, println not run

(if (zero? (random 2)) #true "hi")
; - : (U String True)  ; U for union type
; type is determined before running


(ann #true Boolean)
(ann #true (U String Boolean))
(ann #false (U String Boolean))
(ann 123 Any)
; annotate a type



; Base Types.
; Any
;   Boolean
;     True
;   String
;   Number
; From a set point of view, True ⊂ Boolean ⊂ Any.
; True = {#true}
; Boolean = {#true, #false}
;
; U: a type constructor.
;  Constructs types from types
; (U T ...)
; (U) = {}

(define-type T (U String Number))


; -> : a type constructor, for function types
(ann not (Any -> Boolean))
; - : (-> Any Boolean : False)


(λ (x y) (if (and (string? x) (boolean? y))
             (list x y)
             #false))
; - : (-> Any Any (U (List String Boolean) False))
; ad hoc types... static typing getting huge in typescript
; classical static typing, return type in if branches have to be same
; return type is union type


string?
; - : (-> Any Boolean : String)


(struct Success-Real ([result : Real]) #:transparent)
(struct Failure () #:transparent)
; Failure is a type now at compile time
; Failure is a function at runtime
; Success-Real is a type
; Success-Real is a function

#; (Failure)
; - : Failure
; #<Failure>
#; Failure
; - : (-> Failure)
; #<procedure:Failure>

; compile name type
; Maybe-Real is an interface with 2 subclasses 
(define-type Maybe-Real (U Success-Real Failure))

(ann (Success-Real 123.45) Maybe-Real)

; declaring type of identifier
(: ÷ : (Real Real → Maybe-Real))
(define (÷ x y)
  (if (zero? y)
      (Failure)
      (Success-Real (/ x y))))


; define with pattern matching
(: div : (Real Real → Maybe-Real))
(define/match (div x y)
  [(_ 0) (Failure)]
  [(_ _) (Success-Real (/ x y))])

(: maybe-add-3 : Maybe-Real → Maybe-Real)
(define/match (maybe-add-3 r)
  [((Failure)) (Failure)]
  [((Success-Real x)) (Success-Real (+ x 3))])

(maybe-add-3 (div 4 2))
; - : Maybe-Real
; (Success-Real 5)

(maybe-add-3 (div 2 0))
; - : Maybe-Real
; (Failure)

; Propagates Failure ....

#; (Boxof (U Real String))
#; (Boxof Real)

(: b (Boxof (U Real)))
(define b (box 123))

(: f : (Boxof (U Real String)) → Void)
(define (f b)
  (set-box! b "string")
  (void))
; once there is mutation, typing is difficult

#; 
(f b)






