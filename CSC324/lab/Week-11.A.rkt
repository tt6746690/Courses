#lang typed/racket #| CSC324 2017 Fall Lab : Algebraic Datatypes |#

; Notice the ‘#lang typed/racket’ above.

; Let's stick to non-imaginary numbers.
(require/typed typed/racket
               [+ (Real * → Real)]
               [- (Real Real * → Real)]               
               [* (Real * → Real)]
               [/ (Real Real * → Real)]
               [sqr (Real → Real)]
               [abs (Real → Real)])

#| Algebraic Datatypes

 A struct/record type with fields of type T1, ..., Tn is considered a “product” of the types Ti:
  it's essentially the cartesian product T1 × ⋯ × Tn of the types.

 A union of types T1, ..., Tn is considered a “sum” of the types Ti.

 Roughly, in OO terms:
   a class with instance variables of type T1, ..., Tn is a product of the types Ti
   an interface with classes T1, ..., Tn implementing it is a sum of the types Ti

 Classically, unlike being able to add new implementations in OO, algebraic sums are closed:
   there is a fixed set of types forming the sum

 But functions of the type, unlike methods in OO, are open:
   new functions can be added by specifying the behaviour on each of the fixed set of subtypes |#

; Defining new typed structs.
#;(struct <struct-id> ([<id> : <Type>] ...) #:transparent)

; For example:
(struct Point ((x : Real) (y : Real)) #:transparent)

; That makes a new compile-time type Point, and run-time functions:
Point ; Constructor.
Point? ; Type Predicate.
Point-x ; Field Accessor.
Point-y ; Fiele Accessor.
; Note their types when printed in the Interactions.

; The keyword #:transparent makes, among other things, a printed representation showing the fields.
(Point 3 4) ; Prints as (Point 3 4), instead of the opaque representation #<Point>.

; Compile-time operation ‘ann’ takes an expression and checks that it is compatible with a given type.
; It's essentially a compile-time assertion.
#;(ann <expr> <Type>)

(ann (Point 3 4) Point)
; Notice the first ‘Point’ is the runtime variable referring to a function,
;  and the second ‘Point’ is the compile-time type.

(ann Point (Real Real → Point))

; Alternative notations for function types:
(ann Point (Real Real -> Point)) ; Both kinds of arrow work.
(ann Point (→ Real Real Point)) ; Can be used prefix.
(ann Point (-> Real Real Point))

; Two more product types:
(struct Circle ((centre : Point) (radius : Real)) #:transparent)
(struct Rectangle ((corner-1 : Point) (corner-2 : Point)) #:transparent)

; A sum type of those two types:
(define-type Shape (U Circle Rectangle))

(define p1 (Point 3 4))
(define p2 (Point 10 20))
(define c (Circle p1 5))
(define r (Rectangle p1 p2))
(ann c Circle)
(ann c Shape)
(ann r Rectangle)
(ann r Shape)

; Naming the union of the two types used:
#;(define-type <id> <Type>) ; Name a type at compile time.
#;(U <Type> ...) ; Union of types.

; A runtime identifier can be declared to be of a certain type, which is checked at compile time
;  similarly to ‘ann’, and the identifier is treated as that type.
#;(: <id> : <Type>)

(: c1 : Shape)
(define c1 c)
#;(Circle-radius c1) ; Doesn't type-check.

(: diameter : Circle → Real)
(define (diameter c)
  ; A struct type creates a new pattern that match recognizes:
  (match c [(Circle _ r) (* 2 r)]))

#| ★ Declare and implement ‘area’ that takes a Shape and returns its area.
 Use pattern matching, rather than type predicates and accessors, to determine which type of shape
  and access its components. |#
 

(: area : Shape → Real)
(define (area s)
  (match s
    [(Circle _ r) (* pi (* r r))]
    [(Rectangle p1 p2) (abs (* (- (Point-x p1) (Point-x p2))
                               (- (Point-y p1) (Point-y p2))))]))


#| ★ Define three new types Tree, Empty, and Node.

 Tree represents a Real-labelled binary tree, terminated by empty nodes.

 Tree is either Empty or a Node.
 Empty has no fields.
 Node has a Real and two Trees.

 Declare and implement ‘tree-sum’ that sums all the Reals in the nodes of a Tree. |#


(struct Empty () #:transparent)
(struct Node ((value : Real) (left : Tree) (right : Tree)) #:transparent)
(define-type Tree (U Empty Node))


(define t (Node 5
                (Node 5
                      (Empty)
                      (Node 3 (Empty) (Empty)))
                (Node 5
                      (Node 1 (Empty) (Empty))
                      (Node -10 (Empty) (Empty)))))

(: tree-sum : Tree → Real)
(define (tree-sum t)
  (match t
    [(Node v l r) (+ (+ v (tree-sum l)) (tree-sum r))]
    [(Empty) 0]))

