;; The first three lines of this file were inserted by DrRacket. They record metadata
;; about the language level of this file in a form that our tools can easily process.
#reader(lib "2017-fall-reader.rkt" "csc104")((modname w202) (compthink-settings #hash((prefix-types? . #f))))


(define (flatten anything)
  (cond [(list? anything) (apply append
                                 (map flatten anything))]
        [else (list anything)]))

(require (only-in 2htdp/batch-io read-xexpr/web))
(define html (read-xexpr/web "http://www.english.utoronto.ca/about.htm"))

(map odd? (list 3 2 4))
; (list #true #false #false)

; filter predicate list
; equivalent to SELECT from SQL
(filter odd? (list 3 2 4))
; (list 3)





; filters out long strings
(define (long-string? part)
  (and (string? part)
       (> (string-length part) 250)))
(filter long-string? (flatten html))



; string append
(string-append "1" "2" "3")
; "123"


; append all to a string
(apply string-append (filter long-string? (flatten html)))


; add a space between string during concatenation
(define (add-space a-string)
  (string-append a-string " "))
(apply string-append
       (map add-space
            (filter long-string? (flatten html))))

; note result is listed first here
; bullet point give hints to the big picture, the result of evaluation
; if interested in detail, can look at neseted func calls


; local
; makes temporary definitions, expression local to 'local'
; not a function call
; [] means a grouping
(local [(define x 123)]
  123)

(local [(define (long-string? part)
          (and (string? part)
               (> (string-length part) 250)))]
  (long-string? "abc"))


; using function long-string? with html and flatten
(local [(define (long-string? part)
          (and (string? part)
               (> (string-length part) 250)))]
  (filter long-string? (flatten html)))



; local function local to function definition
; local function dependent on parameter type
(define (long-strings a-list how-long)
  (local [(define (long-string? part)
            (and (string? part)
                 (> (string-length part) how-long)))]
    (filter long-string? a-list)))

; filter out list of string with length > 2
; note local defines are carried along operations
(steps (long-strings (list "hi" "hey" "ho") 2))
;★ steps ★
;★ (long-strings (list "hi" "hey" "ho") 2)
;• (local [(define (long-string? part)
;            (and (string? part) (> (string-length part) 2)))]
;    (filter long-string? (list "hi" "hey" "ho")))
;• (local [(define (long-string? part)
;            (and (string? part) (> (string-length part) 2)))]
;    (list "hey"))
;• (list "hey")

; scope
; determined lexically/statically, based on place of code










