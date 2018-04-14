Initial commit.





[java basics tutorial point](https://www.tutorialspoint.com/java/java_overview.htm)


Java cd
+ access modifiers
+ non-access modifiers

Java Variables
+ Local Variables
  + inside methods, constructors, blocks and destroyed afterwards
  + cant be used with access modifier  
+ class variables (static)
  + declared in class, outside any method, with `static` keyword.
  + only one copy of each class variable per class, regardless how many objects created from it.
  + created when program starts and destroyed when program stops
+ Instance variables (non-static)
  + within a class but outside any method
  + created when class is instantiated and destroyed when object is destroyed
  + can be accessed from any method, constructor or blocks of that particular class
  + access modifiers can be added
  + have default values
  + can be accessed directly by calling the variable name within the class


Java Enums
  + restrict variable to one of few predefined values


Datatypes
+ primitive data types  
  + byte
    + 8 bit
  + short
    + 16 bit
  + int
    + 32 bit
  + long
    + 64 bit
  + float
    + 32 bit floating point
  + double
    + 64 bit floating point
  + boolean
    + one bit of info  
  + char
    + a single 16 bit UNICODE
+ reference/object data types
  + created using constructors
