#| Filtering |#

(define database (list (list "Ada Lovelace"  1815  .)
                       (list "Alan Turing"   1912  .)
                       (list "Alonzo Church" 1903  .)
                       (list "Grace Hopper"  1906  .)
                       (list "Miriam Mann"   1907  .)))


(define year second)
(define name first)

(define (born-after-1905? a-person)
  (> (year a-person) 1905))

; (map string-append ", " (map name (filter born-after-1905? database)))


; partial design: manipulation on arguments
(check-expect (fix-2nd string-append ", ")
              (local [(define (string-append2 1st)
                        (string-append 1st ", "))]
                string-append2))

; higher-order-function: take and/or produces function(s) as argument(s) or result
; basically fix 2nd argument of f when calling (fix-2nd f foo) 
(define (fix-2nd f 2nd)
  (local [(define (f2 1st)
            (f 1st 2nd))]
    f2))
; Note f2 is a local function that is returned
; f2 has a closure,
;   parameters, other names from outer scope
; functions are values, constructed during definition, grabs definition
; from the outer space
; if fix-2nd called twice with same arguments, we return function by value
; that are separate from each other, they have different behavior


(map (fix-2nd string-append ", ") (map name (filter born-after-1905? database)))
