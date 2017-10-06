;; The first three lines of this file were inserted by DrRacket. They record metadata
;; about the language level of this file in a form that our tools can easily process.
#reader(lib "2017-fall-reader.rkt" "csc104")((modname Week-3) (compthink-settings #hash((prefix-types? . #f))))
#| CSC324 2017F Lab
   ================

 Below you create some common Higher Order Functions: functions that take and/or produce functions.

 In particular, most of the functions in this lab both take and produce functions, and support a
  "point free" style of programming where one thinks in terms of composing functions. This is a
  common style in functional programming.

 Briefly, it means preferring composition (f∘g)(x) over application f(g(x)).
 The term "point free" comes from being able to express a function without directly expressing what
  it does to a dummy "point". For example: let f = sin ∘ cos, versus let f(x) = sin(cos(x)). |#


#| ★ Implement mapply ★

 This is a common variant of map.
 You will also re-implement it using one of the other HOFs below, to compare the styles. |#

(check-expect (mapply + (list (list 1 2 3)
                              (list 40 50)
                              (list 600 700 800)))
              (list 6 90 2100))

; ★ First, fix this, and make a fuller or full design from it:
(check-expect (mapply + (list (list 1 2 3)
                              (list 40 50)
                              (list 600 700 800)))
              ; The "′" character is typed by typing \prime [without a space after] and then
	      ; Alt-\ or Ctrl-\ .
              (local [(define (+′ a-list) (apply + a-list))]   
                (list (+′ (list 1 2 3))
                      (+′ (list 40 50))
                      (+′ (list 600 700 800)))))

; ★ First, fix this, and make a fuller or full design from it:
(check-expect (mapply + (list (list 1 2 3)
                              (list 40 50)
                              (list 600 700 800)))
              ; The "′" character is typed by typing \prime [without a space after] and then
	      ; Alt-\ or Ctrl-\ .
             (local [(define (+′ a-list) (apply + a-list))]  
                (map +′ (list (list 1 2 3)
                                    (list 40 50)
                                    (list 600 700 800)))))


#;(define (mapply f a-list)
  (local [(define (accu a-list) (apply f a-list))]
    (map accu a-list)))


#| ★ Recall from Lecture ★ |#

(check-expect (map (fix-2nd string-append ", ")
                   (list "rose" "amy" "clara" "bill"))
              (list "rose, " "amy, " "clara, " "bill, "))

; The csc104 language syntactically forbids computing a function the function position of a call:
#;((fix-2nd string-append ", ") "rose")
; That's to catch beginner errors: we don't *write* HOFs in CSC 104, although we *use* them early and
;  extensively in that course.

; But computing and passing a function to another HOF [e.g. map in the first check-expect above]
;  is allowed, as is naming it then calling it:
(check-expect (local [(define comma-space (fix-2nd string-append ", "))]
                (comma-space "jodie"))
              "jodie, ")

; Comparing functions for equality is disallowed, because it's undecidable [e.g. the Halting Problem].
; idea is cant compare functions
#;(check-expect (fix-2nd string-append ", ")
                (local [(define (string-append′ 1st) ; That's the prime character: \prime alt-\ or ctrl-\ .
                          (string-append 1st ", "))]
                   string-append′))

; But that commented-out check-expect expresses a full design:
(define (fix-2nd f 2nd)
  (local [(define (f′ 1st)
            (f 1st 2nd))]
    f′))

; Stepping a call to fix-2nd produces an error since there isn't a good expression to reduce it to:
#;(step (fix-2nd string-append ", "))
; That motivates moving to a larger language with anonymous functions, later this week.


#| ★ Implement fix-1st ★ |#

(check-expect (map (fix-1st rotate 45) (list (triangle 40 "solid" "maroon")
                                             (square 30 "outline" "navy")))
              (list (rotate 45 (triangle 40 "solid" "maroon"))
                    (rotate 45 (square 30 "outline" "navy"))))

(define (fix-1st f 1st)
  (local [(define (f′ 2nd)
            (f 1st 2nd))]
    f′))


; ★ Then re-implement mapply using fix-1st.
; this is like composing functions f dot g
(define (mapply f a-list)
    (map (fix-1st apply f) a-list))

#| ★ Implement ¬ [negation] ★

 To type the "¬" character: type \neg and then alt-\ or ctrl-\ . |#


(check-expect (filter (¬ list?) (list 1 (list 2 3) 4 5 (list)))
              (list 1 4 5))

#;(check-expect (¬ list?)
                (local [(define (¬list? v)
                          (not (list? v)))]
                  ¬list?))

; (define (¬ f)
;   (local [(define (¬f arg)
;             (not (f arg)))]
;     ¬f))


#| ★ Implement ∘ [compose] ★

 To type the "∘" character: type \circ and then alt-\ or ctrl-\ . |#

(check-expect (local [(define -sqr (∘ - sqr))]
                (-sqr 3))
              -9)

(define (∘ f g)
  (local [(define (compose arg)
            (f (g arg)))]
    compose))

; ★ Then re-implement ¬ using ∘ and not [without fix-1st].

; (define (¬ f)
  ; (∘ not f))

; ★ Then re-implement ¬ using fix-1st, ∘, and not.

(define ¬
  (fix-1st ∘ not))

; a point free function that uses a non-point free function (fix-1st)
