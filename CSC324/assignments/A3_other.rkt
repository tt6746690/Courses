#lang racket 
; The following all produce the same value:
#;(string->symbol "Boolean") ; Symbol with the name Boolean
#;'Boolean ; Symbol with the name Boolean
#;`Boolean ; Symbol with the name Boolean
; Symbol with the name Boolean
#;(local [(define b 'Boolean)]
    b)
; Symbol with the name Boolean
#; (local [(define b 'Boolean)]
     `,b)
#; (first '(Boolean)) ; Symbol with the name Boolean
#; (string->symbol (string-append "B" "oolean")) ; Symbol with the name Boolean

; The following all produce the same value:
; (list 'Boolean 'Boolean '→ 'Boolean)
; '(Boolean Boolean → Boolean)
; `(Boolean Boolean → Boolean)
; (list* 'Boolean '(Boolean → Boolean))
; (define b 'Boolean)
; (append (list b b) `(,(string->symbol "→") ,b))



#; (symbol? Boolean) ; Boolean doesn't even refer to any value.
#; (symbol? (string->symbol (number->string 324))) ; #true
; True. But not because it's quoted. The value isn't quoted. The value also isn't unquoted.
; The question of whether a value is quoted or unquoted is a meaningless question.
; To use a famous example from linguistics: do green ideas sleep furiously? No? Yes? Neither? It's not a valid question.
; (list? '(λ ((x : Boolean)) 1)) ; Yes, because the code '(_) makes a *list* *value* at runtime. But not because it's a quoted list: the phrase "quoted list" is meaningless.
; (list?  (λ ((x : Boolean)) 1))

; "abc"
; Is the value produced by that quoted? That's a meaningless question.
; If someone refers to a string, when referring to code, they must mean the runtime value.
; So: is that string quoted? That's a meaningless question.

; (define s "abc")
; s ; Is this string quoted? That's a meaningless question.
