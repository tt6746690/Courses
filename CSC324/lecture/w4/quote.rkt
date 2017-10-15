#lang racket #| Quotation. |#

#| Forms other than identifiers can be quoted.

 For boolean, string, and numeric literal forms, quotation has no effect. |#

'324  ; The *number* 324.
'-324 ; The *number* -324.
'"abc" ; The *string* "abc".
'#true ; The *boolean* #true.

#| Lisp needed to represent nested structure, in order to represent Lisp code.

 They chose heterogeneous lists as the datatype, nesting them for nested structure,
  and so made quotation of parenthesized forms create nested lists, recursively
  quoting the components. |#

(define one 1)

(list (+ 1 2) (+ one 2) one)
(list 3 (+ 1 2) 1)
(list 3 3 1)

; Excercise. Predict and try:
#;((+ 1 2) (+ one 2) one)

(quote
 ((+ 1 2)
  (+ one 2)
  one))
; '((+ 1 2) (+ one 2) one)

(list
 (quote (+ 1 2))
 (quote (+ one 2))
 (quote one))
; (list '(+ 1 2) '(+ one 2) 'one) WRONG!
;'((+ 1 2) (+ one 2) one) GOOD!

(list
 (list (quote +)
       (quote 1)
       (quote 2))
 (list (quote +)
       (quote one)
       (quote 2))
 (quote one))
; '((+ 1 2) (+ one 2) one)

(list
 (list (quote +)
       1
       2)
 (list (quote +)
       (quote one)
       2)
 (quote one))
;'((+ 1 2) (+ one 2) one)
; makes sense since quote on boolean/string/number literals are boolean/string/literal

; For #lang racket, DrRacket prints that as:
'((+ 1 2)
  (+ one 2)
  one)
; To see it as calls to the list constructor: go to the Language dialog and click the
;  "Show Details" button, set "Output Style" to "Constructor", and run this again.


; after list ctor
#; (list (list '+ 1 2) (list '+ 'one 2) 'one)

; Read the introduction and "Quote" section of "Intermezzo 2: Quote, Unquote" from:
;   http://www.ccs.neu.edu/home/matthias/HtDP2e/Draft/i2-3.html

; The algebraic rewrite rules for quote are:
;
#;(quote (<part>
          ...))
; →
#;(list (quote <part>)
        ...)
;
#;(quote <boolean/string/numeric-literal>)
; →
#;<boolean/string/numeric-literal>
;
#;(quote <identifier>)
; No rewrite: it is a **symbol literal**

; Exercise. Write out the steps for:
#;'(define (double x) (* 2 x))

; (list 'define '(define x) '(* 2 x))
; (list 'define (list 'double 'x) (list '* 2 'x)) GOOD!
