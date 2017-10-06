#| Filtering |#

(define database (list (list "Ada Lovelace"  1815  .)
                       (list "Alan Turing"   1912  .)
                       (list "Alonzo Church" 1903  .)
                       (list "Grace Hopper"  1906  .)
                       (list "Miriam Mann"   1907  .)))

#;(list? database)

#| Three element accessors in csc104:
     • (first  (list a b c ...)) -> a
     • (second (list a b c ...)) -> b
     • (third  (list a b c ...)) -> c |#

(check-expect (first (list 3 2 4))
              3)
(check-expect (second (list 3 2 4))
              2)
(check-expect (third (list 3 2 4))
              4)

#;(list? (first database))
#;(second database)
#;(first (third database))

(steps parallel
       (map first database))

(define (born-after-1906? a-person)
  (> (second a-person) 1906))

(steps (born-after-1906? (second database)))

(steps parallel
       [hide born-after-1906?]
       (map born-after-1906? database))

#| Filtering

 A common operation on collections.

 (filter ϕ ℓ) is the list of elements from list ℓ for which predicate ϕ is true.

 It's "filter in", not "filter out".

 You can imagine an implicit map of the predicate, which is then used to select elements.

 On tests, during calculation you may cross out the elements for which the the predicate is false,
  or underline or circle elements for which it is true [with the latter approach don't forget that
  the list constructor gets carried forward].

 Databasing.
   Relational algebra: σ_ϕ(R) = (filter ϕ R)
   SQL SELECT.

 Comprehensions.
   Math/Theory {x ∈ D : ϕ(x)} ≈ (filter ϕ D)
   Python [e for e in L if ϕ(e)] = (filter ϕ L) |#

(filter born-after-1906? database)
