
Java Array  

+ a fixed-size sequential collection of elements of the __same__ type.
+ object not primitive
+ Initialization
  + Creating arrays
    + `nameofArray = new TypeOfElements[arrayLength]`  
  + Declare array variables
    + `dataType[] arrayRefVar;`, i.e. `String[] alphabet = {"a", "b", "c"}`   
+ multidimensional array
+ does not provide conveniences such as the ability to grow or sort elements


Java collections


Generics
+ way of extending static typing to classes when the exact type of data the classes will operate on is unknown.
+ used to circumvent Java's rule of declaring type of variable before instantiation.
+ eg
  + Want a `List` containing type `E`, where `E` is any class/interface. calling the `get` method on `List` return objects of type `E`  
  + `Map` interface is an example where two generic types need to be specified
    + one for keys
    + one for values

+ `List<E>` means programmer should replace `E` with same data type every time it appears
  + Any list will have elements of one Class/Interface, i.e. `String`
  + `List<String> strs = new ArrayList<String>();`   
  + `String s = strs.get(0)`
  + here we create an `ArrayList` of `Strings` and access an element  

+ We write `List<String>` instead of `ArrayList<String>` better
