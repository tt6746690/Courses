#| CSC 104 Test 2 Recursion Tracing 2016-2017 |#

#| 2017 Winter ~10 minutes |#

; Assume function ‘r’ has been defined:
(define (r L)
  (cond [(= (length L) 1) L]
        [else (list* (first L) (r (reverse (rest L))))]))

; Reminders.
(check-expect (rest (list 2 0 1 7)) (list 0 1 7))
(check-expect (list* 2 (list 0 1 7)) (list 2 0 1 7))

; Show the the final result value of the following expression:
(r (list 1))

; (cond [(= (length (list 1)) 1) (list 1)]
;       [else (list* (first (list 1)) (r (reverse (rest (list 1)))))])
; (cond [(= 1 1) (list 1)]
;       [else (list* (first (list 1)) (r (reverse (rest (list 1)))))])
; (cond [#true (list 1)]
;       [else (list* (first (list 1)) (r (reverse (rest (list 1)))))])
; (list 1)


; For each of the following expressions, show [at least] the non-cond expressions up to
;  entering the first recursive call and after that call finishes.
; You may abbreviate 'reverse' as 'rev'.
#;(r (list 1 2))

; (list* (first (list 1 2)) (r (reverse (rest (list 1 2)))))
; (list* 1 (r (reverse (list 2))))
; (list* 1 (r (reverse (list 2))))
; (list* 1 (r (list 2)))
; (list* 1 (list 2))
; (list 1 2)

#;(r (list 1 2 3))

; (list* (first (list 1 2 3)) (r (reverse (rest (list 1 2 3)))))
; (list* 1 (r (list 3 2)))
; (list* 1 (list 3 2))
; (list 1 3 2)

#;(r (list 1 2 3 4))

(list* (first (list 1 2 3 4)) (r (reverse (rest (list 1 2 3 4)))))
(list* 1 (r (list 4 3 2)))
(list* 1 (list 4 2 3))
(list 1 4 2 3)

; In other words, show [at least] the non-cond expressions of:
(define rev reverse)
(define (R L)
  (cond [(= (length L) 1) L]
        [else (list* (first L) (r (rev (rest L))))]))
#;(step parallel
        [hide r]
        (R (list 1 2))
        (R (list 1 2 3))
        (R (list 1 2 3 4)))

#| 2016 Fall ~9 minutes |#

; Assume function 'g' has been defined:
(define (g a-string)
  (cond [(< (string-length a-string) 2) a-string]   ; < 2
        [else (string-append (substring a-string 0 2)   ; first 2 chars
                             (g (substring a-string 2 (string-length a-string)))  ; the rest
                             (substring a-string 0 2))]))  ;first 2 chars again

; Reminders.
(check-expect (substring "abcde" 0 2) "ab")
(check-expect (substring "abcde" 2 5) "cde")

; Show the the final result values of the following expressions:
(g "ab")

; "abab"

(g "abc")

; "abcab"

; For the following expression, show [at least] the non-cond expressions up to entering
;  the first recursive call and after that call finishes.
; You may abbreviate 'string-length' as 's-l', 'string-append' as 's-a' and 'substring' as 'sub'.
#;(g "abcde")

; In other words, show [at least] the non-cond expressions of:
(define s-l string-length)
(define s-a string-append)
(define sub substring)
(define (G a-string)
  (cond [(< (s-l a-string) 2) a-string]
        [else (s-a (sub a-string 0 2)
                   (g (sub a-string 2 (s-l a-string)))
                   (sub a-string 0 2))]))
#;(step parallel
        [hide g]
        (G "abcde"))

; (s-a (sub "abcde" 0 2) (g (sub "abcde" 2 (s-l "abcde"))) (sub "abcde" 0 2))
; (s-a "ab" (g (sub "abcde" 2 5)) "ab")
; (s-a "ab" (g "cde") "ab")
; (s-a "ab" "cdecd" "ab")
; (s-a "abcdecdab")

#| 2016 Winter ~9 minutes |#

; Assume function 'f' has been defined:
(define (f a-list)
  (cond [(= (length a-list) 1) a-list]
        [else (append (list (first a-list))
                      (f (rest a-list))
                      (list (first a-list)))]))

; Show the final result values of the following expressions:
#;(f (list "A"))

; (list "A")

#;(f (list "A" "B"))

(list "A" "B" "A")

; For the following expression, show [at least] the non-cond expressions up to entering
;  the first recursive call and after that call finishes.
; You may abbreviate 'length' as 'len' and 'append' as 'app'.
#;(g "abcde")

; In other words, show [at least] the non-cond expressions of:
(define app append)
(define (F a-list)
  (cond [(= (length a-list) 1) a-list]
        [else (app (list (first a-list))
                   (f (rest a-list))
                   (list (first a-list)))]))
#;(step parallel
        [hide f]
        (F (list "A" "B" "C")))

; (append (list (first (list "A" "B" "C"))) (f (rest (list "A" "B" "C"))) (list (first (list "A" "B" "C"))))
; (append (list "A") (f (rest (list "A" "B" "C"))) (list "A"))
; (append (list "A") (f (list "B" "C")) (list "A"))
; (append (list "A") (list "B" "C" "B") (list "A"))
; (list "A" "B" "C" "B" "A")
