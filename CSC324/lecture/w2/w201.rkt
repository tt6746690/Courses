;; The first three lines of this file were inserted by DrRacket. They record metadata
;; about the language level of this file in a form that our tools can easily process.
#reader(lib "2017-fall-reader.rkt" "csc104")((modname w201) (compthink-settings #hash((prefix-types? . #f))))
#| Test-Driven Development, and Recursion on Hierarchical Data |#

; Let's write a function to take a list and produce a list of all "leaf" elements, i.e. all non-list
;  elements from which it's built.

; For example [i.e, a test case]:
(check-expect (flatten (list (list "cat"
                                   "dog"
                                   "snail")
                             (list "rhino"
                                   "hippo")
                             (list)
                             (list "dolphin")))
              (list "cat"
                    "dog"
                    "snail"
                    "rhino"
                    "hippo"
                    "dolphin"))


(check-expect (append (list 1 2) (list 3 2 4) (list 5))
              (list 1 2 3 2 4 5))

; hold down alt/option while moving with arrows keys
; that move by expression


; partial design 
(check-expect (flatten (list (list "cat"
                                   "dog"
                                   "snail")
                             (list "rhino"
                                   "hippo")
                             (list)
                             (list "dolphin")))
              ; partly based on manual inspection and work
              (append (list "cat"
                    "dog"
                    "snail"
                    "rhino"
                    "hippo"
                    "dolphin")))

; still a partial design :handles single depth nested lists
; add apply to append 
(check-expect (flatten (list (list "cat"
                                   "dog"
                                   "snail")
                             (list "rhino"
                                   "hippo")
                             (list)
                             (list "dolphin")))
              ; partly based on manual inspection and work
              (apply append (list (list "cat"
                                   "dog"
                                   "snail")
                             (list "rhino"
                                   "hippo")
                             (list)
                             (list "dolphin"))))


; still a partial design, have to take care of sub-levels with flatten
(check-expect (flatten (list (list "bat"
                                   "rat")
                             (list (list "cat"
                                         "dog"
                                         "snail")
                                   (list "dolphin"))))
              ; partly based on manual inspection and work
              (apply append (list (flatten (list "bat"
                                                 "rat"))
                                  (flatten (list (list "cat"
                                                       "dog"
                                                       "snail")
                                                 (list "dolphin"))))))


; only state is random ...
; functions are pure without side-effect


; equivalent expression with map
(check-expect (flatten (list (list "bat"
                                   "rat")
                             (list (list "cat"
                                         "dog"
                                         "snail")
                                   (list "dolphin"))))
              ; partly based on manual inspection and work
              (apply append (map flatten (list (list "bat"
                                                     "rat")
                                               (list (list "cat"
                                                           "dog"
                                                           "snail")
                                                     (list "dolphin"))))))

; check for base case
(check-expect (flatten (list (list "bat"
                                   "rat")
                             (list "mouse")
                             (list (list "cat"
                                         "dog"
                                         "snail")
                                   (list "dolphin"))))
              ; partly based on manual inspection and work
              (apply append (list (flatten (list "bat"
                                                 "rat"))
                                  (flatten "mouse")
                                  (flatten (list (list "cat"
                                                       "dog"
                                                       "snail")
                                                 (list "dolphin"))))))


;another base case 
(check-expect (flatten "mouse") (list "mouse"))



; append on singleton list produce lists of items, not lists of lists of size 1
(append (list "bat") (list "cat"))
; (list "bat" "cat")

; first somelist -> first item in list



; LoL: lists of lists, can be infinitly nested
; define (flatten LoL)
;  (list)

(define (flatten anything)
  (cond [(list? anything) (apply append
                                 (map flatten anything))]
        [else (list anything)]))

; can be another base case
; where we have (list) as argument to flatten

;(define (flatten anything)
;  (cond [(empty? anything) anything]
;        [(list? anything) (apply append
;                                 (map flatten anything))]
;        [else (list anything)]))


; some use case
(steps parallel
      (flatten "cat"))
; ★ steps ★
; ★ (flatten "cat")
; • (cond [(list? "cat") (apply append (map flatten "cat"))]
;        [else (list "cat")])
; • (cond [#false (apply append (map flatten "cat"))]
;        [else (list "cat")])
; • (cond [else (list "cat")])
; • (list "cat")


; expect to return (list) identity
; relies on the fact that for (map fn (list)) map does not call fn at all
(steps parallel
       (flatten (list)))
;★ steps ★
;★ (flatten (list))
;• (cond [(list? (list)) (apply append (map flatten (list)))]
;        [else (list (list))])
;• (cond [#true (apply append (map flatten (list)))]
;;        [else (list (list))])
;• (apply append (map flatten (list)))
;• (apply append (list))
;• (append)
;• (list)


(steps parallel
       (hide (flatten "cat") (flatten "dog"))
       (flatten (list "cat" "dog")))

;★ steps ★
;★ (flatten (list "cat" "dog"))
;• (cond [(list? (list "cat" "dog")) (apply append (map flatten (list "cat" "dog")))]
;        [else (list (list "cat" "dog"))])
;• (cond [#true (apply append (map flatten (list "cat" "dog")))]
;;        [else (list (list "cat" "dog"))])
;• (apply append (map flatten (list "cat" "dog")))
;• (apply append (list (flatten "cat") (flatten "dog")))
;;• (apply append (list (list "cat") (list "dog")));;
;• (append (list "cat") (list "dog"))
;• (list "cat" "dog")


(steps parallel
       (hide (flatten (list "cat" "dog"))
             (flatten (list "rat" "bat" "snail")))
       (flatten (list (list "cat" "dog")
                      (list "rat" "bat" "snail"))))



; all special semantics; 
; and
; or 
; cond
; define
; check-expect
; step + parallel/hide


; function calls in most languages are eager
; In eager evaluation, an expression is evaluated as soon as it is bound to a variable. 

; programming languagge
; syntax: leaning syntax is important but 
; semantics: meaning of operation that language provides, how things work!






















