

+ s expression (sexprs)
    + recursive definition, is 
        + an atom, or
        + an expression of the form `(x . y)`

+ de bruijin index
    + each de bruijin index is a natural number representing an occurence of a variable in a lambda term, denoting the number of binders in scope between that occurence and its corresponding binder


+ free variables 
    + refer to variables used in a function that are neither local variables nor parameters of that function


+ combinators 
    + a combinator is a higher-order function that uses only function application and earlier defined combinators to define a result from its argument
    + read more on [lambda calculus](https://en.wikipedia.org/wiki/Combinatory_logic#Combinatory_calculi) and [combinatory calculi](https://en.wikipedia.org/wiki/Combinatory_logic#Combinatory_calculi)