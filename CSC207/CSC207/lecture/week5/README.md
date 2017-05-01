
__CRC__

[good link](http://agilemodeling.com/artifacts/crcModel.htm)  

A __Class__ represents a collection of similar objects. An object is a person, place, thing, event, or concept that is relevant to the system at hand

A __Responsibility__ is anything that a class knows or does. For example, students have names, addresses, and phone numbers.  

Sometimes a class has a responsibility to fulfill, but not have enough information to do it; therefore need to collaborate with other classes.

A __Collaboration__ takes one of two forms: A request for information or a request to do something. - See more at: http://agilemodeling.com/artifacts/crcModel.htm#sthash.dGfWYgyR.dpuf


__Interface__

In the Java programming language, an interface is a __reference type__, similar to a class, that can contain only constants, method signatures, default methods, static methods, and nested types. _Method bodies exist only for default methods and static methods._ Interfaces cannot be instantiated—they can only be implemented by classes or _extended by other interfaces_. They essentially provide an API

```java
public interface OperateCar {

   // An enum with values RIGHT, LEFT
   int turn(Direction direction,
            double radius,
            double startSpeed,
            double endSpeed);

  // method signatures
   int changeLanes(Direction direction,
                   double startSpeed,
                   double endSpeed);
}
```

When an instantiable class implements an interface, it provides __a method body__ for each of the methods declared in the interface

```Java
public class OperateBMW760i implements OperateCar {

    // the OperateCar method signatures, with implementation --
    // for example:
    int signalTurn(Direction direction, boolean signalOn) {
       // code to turn BMW's LEFT turn indicator lights on
       // code to turn BMW's LEFT turn indicator lights off
       // code to turn BMW's RIGHT turn indicator lights on
       // code to turn BMW's RIGHT turn indicator lights off
    }

}

```


Interface content
+ abstract method   
  + follow by `;` instead of braces as there are no implementations
+ Default method  
  + defined with the default modifier
+ static method
  + defined with static keyword
+ constant declaration







__Inheritance__


What you can do in the subclass

+ The inherited fields can be used directly, just like any other fields.
+ You can __declare a field__ in the subclass with the same name as the one in the superclass, thus __hiding__ it (not recommended).
+ You can declare new fields in the subclass that are not in the superclass.
+ The inherited methods can be used directly as they are.
+ You can write a __new instance method__ in the subclass that has the same signature as the one in the superclass, thus __overriding__ it.
+ You can write a __new static method__ in the subclass that has the same signature as the one in the superclass, thus __hiding__ it.  
+ You can declare new methods in the subclass that are not in the superclass.
+ You can write a subclass constructor that invokes the constructor of the superclass, either implicitly or by using the keyword super.  
+ A subclass does not inherit the private members of its parent class.


__Multiple inheritance of implementation__ is the ability to inherit method definitions from multiple classes.  Java prohibits such behavior.  

However it does supports __multiple inheritance of type__, which is the ability of a class to implement more than one interface. _An object can have multiple types: the type of its own class and the types of all the interfaces that the class implements._ This means that if a variable is declared to be the type of an interface, then its value can reference any object that is instantiated from any class that implements the interface.



More on shadowing and overriding

1. An __instance__ method in a subclass with the same signature (name, plus the number and the type of its parameters) and return type as an instance method in the superclass __overrides__ the superclass's method.
2. If a subclass defines a __static__ method with the same signature as a static method in the superclass, then the method in the subclass __hides__ the one in the superclass.



The distinction between hiding a static method and overriding an instance method has important implications:
+ The version of the overridden instance method that gets invoked is the one in the subclass.     
+ The version of the hidden static method that gets invoked depends on whether it is invoked from the superclass or the subclass.   




|  | Variable     | Method |
| :------------- | :------------- |
| Static      | shadow      | shadow |
|Instance | shadow | override |


Shadowing  
+ Basically can access both parent and child class via casting
+ can always cast down...  
+ always call the topmost methods/variables


```java
// s is treated by compiler as Object only
Object s = new String("hi"); // s only has Object method, not String method
System.out.println((String) s.length()); // must cast to
```

Override
+ Only the child class is accessible.




__Abstract class__

Abstract classes are similar to interfaces. You cannot instantiate them, and they may contain a mix of methods declared with or without an implementation. However, with abstract classes, _you can declare fields that are not static and final_, and define _public, protected, and private concrete methods (concrete = implemented)_. With interfaces, all fields are automatically public, static, and final, and all methods that you declare or define (as default methods) are public.



__Autoboxing and unboxing__

__Autoboxing__ is the automatic conversion that the Java compiler makes between the primitive types and their corresponding object wrapper classes. For example, converting an int to an Integer, a double to a Double, and so on. If the conversion goes the other way, this is called unboxing.

```java
Character ch = 'a'; // autoboxing
```


Although you add the int values as primitive types, rather than Integer objects, to li, the code compiles. Because li is a list of Integer objects, not a list of int values, as specified by the generics

```java
List<Integer> li = new ArrayList<>();
for (int i = 1; i < 50; i += 2)
    li.add(i); // compiles to li.add(Integer.valueOf(i));

```

__Unboxing__

Because the remainder (%) and unary plus (+=) operators do not apply to Integer objects, you may wonder why the Java compiler compiles the method without issuing any errors. The compiler does not generate an error because it invokes the `intValue` method to convert an Integer to an int at runtime:

```java
public static int sumEven(List<Integer> li) {
    int sum = 0;
    for (Integer i: li)
        if (i % 2 == 0)
            sum += i;
        return sum;
}
```





__Generics__

In a nutshell, generics enable types (classes and interfaces) to be parameters when defining classes, interfaces and methods. type parameters provide a way for you to re-use the same code with different inputs.

Advantages
1. Stronger type checks at compile time.  
2. Elimination of casts.

```java
List list = new ArrayList();
list.add("hello"); // arraylist only addds objects, autobox happen here
String s = (String) list.get(0); // requires cast to get String Object instead of just Object

List<String> list = new ArrayList<String>();
list.add("hello");
String s = list.get(0);   // no cast
```

3. Enabling programmers to implement generic algorithms.



_Generic version of Class_  

A type variable can be any non-primitive type you specify: any class type, any interface type, any array type, or even another type variable.

```java
/**
 * Generic version of the Box class.
 * @param <T> the type of the value being boxed
 */
public class Box<T> {
    // T stands for "Type"
    private T t;

    public void set(T t) { this.t = t; }
    public T get() { return t; }
}
```


_Invoking and instantiating a generic type_

To reference the generic Box class from within your code, you must perform a generic type invocation, which replaces T with some concrete value, such as Integer:

```java
Box<Integer> integerBox; // declaration; not creating new object, just creating a reference

Box<Integer> integerBox = new Box<Integer>(); // instantiation
Box<Integer> integerBox = new Box<>(); // after java SE 7, if compiler can infer Type
```


__Memory__

Stack
+ used for execution of a thread
+ FILO
+ specific, shortlived values sometimes references to Objects in heap space  

Heap
+ used by Java to allocate memory to Objects
