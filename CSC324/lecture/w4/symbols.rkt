#lang racket
#| Symbol: a Datatype for Names. |#

#| First, choose "The Racket language" from DrRacket's language menu. |#

#| Names.

 In life, including CS discussions and programming, we frequently use names,
  where the only property that matters is whether two names are the same or not.

 Examples:

   html tags
     • "is this a 'div' tag?" : affects the meaning
     • "is this tag three letters long?" : doesn't affect meaning

   program tokens in Python
     • "is this token 'def'" : affects the meaning
     • "is this token letters long?" : doesn't affect meaning in Python |#


#| Names in the first few decades of machine computing.

 Strings were expensive: to store, to compare character by character.
 Most programs were numerical.

 But, interpreters and compilers did not have the luxury of ignoring text,
  because a main point of source code was for humans to be able to describe a
  program via text. In particular, people wanted to use names for operations
  and variables.

 One of the research goals of Lisp was to allow Lisp interpreters, compilers, and
  other programs that work on programs, to be written naturally in Lisp itself.
  So they included a datatype for names, along with a literal notation for them.

 The dual benefit to expressing one's intention as directly as possible in code:
   Help human readers understand and reason about it.
   Help computers     understand and reason about it.
     E.g. check correctness, compile efficiently, improve error message, etc.
  Those are actually the same: humans write the code that reasons about code,
   based on their reasoning abouit code.

 Although Lisp was created to study potential programming languages and paradigms
  unconstrained by 1950s machine resource limitations, a graduate student noticed
  it was actually implementable without much effort [although not useful for the
  simplistic programs of the 1950s, which didn't need the expressiveness and so
  couldn't afford the resource overhead that the first simplistic implementation
  of Lisp had]. In particular, symbols could be hashed for quick comparison.

 See, for example:  https://en.wikipedia.org/wiki/String_interning

 In PL theory this leads to De Bruijn indexing, which replaces names with the
  minimal information needed to locate their bindings:
    https://en.wikipedia.org/wiki/De_Bruijn_index
  Compilers and other program analyzers frequently use that or a similar approach,
   since it literally tracks the minimal necessary information.

 Without a symbol-like datatype, programmers often mimic it with the "named
  constants" design pattern, e.g.
   public final UP = 0;
   public final DOWN = 1;
   public final LEFT = 2;
   public final RIGHT = 3;
  and many programming languages or libraries support this with an "enumerated type"
  feature. |#
   

#| Quoting names in Lisps.

 In Lisps, Schemes, Clojure, and Racket, the following two syntaxes are
  semantically equivalent ways to include a literal a symbol in code:

   '<identifier>
   (quote <identifier>)

 Whitespace and parentheses were already being used to mark up the token and
  tree structure of Lisp code [commas are redundant, and needlessly expensive
  for 1950s machines]. Disallowing them in symbols meant symbols could be
  delimited by a single leading character [an ending delimeter is redundant,
  and needlessly expensive for 1950s machines]. And the main initial purpose
  --- quoting Lisp code itself to turn it into data for interpretation and
  compilation --- would be supported. |#

'hi
(quote hi)
'+
(quote +)
'~!@#$%^&*:<>.?/
(quote ~!@#$%^&*:<>.?/)

; The characters in the name of this variable happen to be the same as the
;  characters in the string which is its value. That is a coincidence, and
;  we expect absolutely *no* consequences of that.
(define abc "abc")

(define xyz 'xyx) ; Coincidence, with no consequences.

(define uvw (quote uvw)) ; Still a coincidence.

(define a 0) ; Absolutely no effect on the values of the following literals:
"a"
'a
(quote a)

; At this point we can treat
;   (quote <identifier>)
; the same way we would treat:
;   "<characters>"
; There is no algebraic rule to evaluate a literal, except the "obvious" rule:
;   don't evaluate any of the characters as expressions

(define (Quote name) (quote name))
; Exercose. Predict and try:
#;(Quote me)
; me: undefined
; cannot reference an identifier before its definition
; since have to bind expression me to a name when calling a function
#;(Quote name)
; name: undefined
;    note the usuall (quote name) works
#;(Quote abc)
; 'name
;    note (define abc "abc"), but the return value is 'name
#;(Quote xyz)
; 'name
#;(Quote a)
; 'name
;   note a is a symbol 

(define (f v) (list v "v" 'v (quote v)))
; Exercise. Predict and try:
#;(f v)
; v: undefined
#;(f a)
; (list  0 "v" 'v 'v)  WRONG!
; '(0 "v" v v)
;    note (define a 0)

; The predicate for the symbol datatype is 'symbol?'.
(symbol? abc) ; For #lang racket, DrRacket prints #true and #false as '#t' and '#f'.
; #f
(symbol? 'abc)
; #t
(symbol? xyz)
; #t (xyz='xyz)
(symbol? a)
; #f  (a=0)
(symbol? #true)
; #f 
