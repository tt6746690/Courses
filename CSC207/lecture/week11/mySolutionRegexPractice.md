

1. a string of digits

`\d`

remember that `\d` only matches to one digit add `+` to match one or more

```
\d+
```


2. only alphanumeric character with length at least 4

```
[0-9a-zA-Z]{4,}
```

3. match string with two words that are the same

```
(\w).*\\1
```
wrong have to think about that comes before the word and adding word boundary.

```
.*\b(\w+)\b.*\1.*
```

4. match a string containing 4 consecutive lowercase consonants

```
.*[^euioa]{4,}.*
```

remember to set the the set of lowercase letters

```
[a-z&&[^euioa]]{4}
```


5. match a person's name, assume to have 2 words, each starting with a capital letter and consisting of entirely of letters

```
([A-Z][a-zA-Z]*)\s+\1
```

This doesnt work because capture group reference to exactly the same characters. So something like `Harry Harry` would pass but `Harry Potter` would not , so instead

```
[A-Z][a-z]+\s+[A-Z][a-z]+
```

6. match a valid postal code `M4R 1E4` (with one space in between)

```
[A-Z][0-9][A-Z]\s[0-9][A-Z][0-9]
```

or solution, note that a single space ` ` works for matching space as well

```
[A-Z]\d[A-Z] \d[A-Z]\d
```


7. match a valid postal code with same digits

```
[A-Z](\d)[A-Z]\s\1[A-Z]\1
```

8. match string to represent time in HH:mm:ss format

```
(0[1-9]|1[0-2]):[0-5][0-9]:[0-5][0-9]
```


9.

```
[A-Z]{3}[0-9]{3}H1\tF\t(Intro\s\w{12})\tLEC\t
```

```
([A-Z]{3}[0-9]{3}H1)\tF\t(Intro\s\w{12})\t LEC \t(([0-9]{4})\t)\4
```

ask about this question...
