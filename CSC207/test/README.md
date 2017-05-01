practice test


1. What are the main components of a class in Java? What are the standard accessibility
modifiers for each (e.g., protected, private, etc.)? When would you want to use
a non-standard accessibility modifier for a variable? for a method? for a constructor?


Constructor,instance variable, methods.

public: accessible everywhere
private: only visible in this class
default: package visible
protected: package and subclasses


2. In what ways are methods similar to constructors? In what ways are they different?
(Consider: the syntax for coding each, how they show up in the Java memory model,
features such as calling other constructors, whether or not Java provides them by
default, returning values, the applicability of terms like “instance” and “static”, etc..)


they both have a name. can be called.

constructor:
+ name restricted to class name.
+ return type always void.
+ called every time upon instantiation.
+ `super()` calls parent constructor, called by default if no `super` mentioned
+ cant be static because constructor's purpose is to to instantiate object
  + practically `this` and `super` are non-static. Declaring a static constructor would render them useless.

methods:
+ no restriction on return modifier
+ no restriction on accessibility modifier
+ could be any name
+ could be static, then the method belongs to the class instead of to the instance




3. Is it possible for a class to have more than one parent class? More than one child
class? To implement more than one interface?


Each class can only have one parent class but may have more than one child class and implement more than one interface


4. In which ways are abstract classes and interfaces similar? In what ways are they
different? List the conditions which are necessary and sufficient for a: (a) class to be
declared abstract and (b) for a method to be abstract.

Both cannot be instantiated directly

Abstract class
+ any one of the method is abstract, then the class is abstract
+ therefore able to contain both abstract and concrete method
+ a method is abstract if it is declared but not implemented. abstract method cannot be instantiated and requires subclass to provide proper implementation.
+ can have either public or protected methods and NO private methods
+ can have static, final, static final field


Interface
+ contains signatures of methods
+ can only have abstract method
+ public method by default; no private methods/fields
+ only static final field by default



5. What do the following keywords mean when used in front of a method: final,
static, abstract.

final
+ method cannot be overriden in subclasses

static
+ make method accessible without an instance

abstract
+ method unimplemented and cannot be called since class cannot be instantiated


![comparison](assets/README-7f194.png)



6. List all of the primitive types. How does the memory model reflect the differences between
primitive types and objects. When are two primitive variables “equal”? When
are two objects “equal”? Why is the word “equal” in quotes? Are all non-primitive
types subclasses of the Object class? What features of class Object did we use most
frequently in the lectures?


Primitive types are literals

1. byte  
  + 8 bit signed so -128 ~ 127 integer
2. short
  + 16 bit signed integer
3. int
  + 32 bit signed integer
4. long
  + 64 bit signed integer
5. float
  + 32 bit floating point
6. double
  + 64 bit floating point
7. boolean
8. char
  + 16 bit Unicode character


Note that every variable is implemented as a location in memory

primitive type
+ the value of the variable is stored in the memory address assigned to the variable
+ fixed memory size


Object (reference type)
+ only stores the memory address of where the object is located (where all details resides) – not the values inside the object.
+ although a variable is fixed sized, the object its referring to is not restricted by memory
+ aliasing could happen, when 2 variable refer to the same underlying Object;



In general,

Object:
+ `==` compares object reference, i.e. if two objects refer to the same memory address
+ `equals(Object)` tests if object is equal to the other object, depending on how equality is defined. Therefore different object (different memory location) will evaluate to be true if what they store are equivalent as defined.

primitives:
+ `==` compares value of the type
+ `equals()` is not defined

Autoboxed primitives (wrapper class)
+ `==` compares object reference
+ `equals()` compares wrapped values


Every non-primitive types are subclasses of the Object class

As for Object class we used
+ `getClass`
+ `toString` in the format of

```java
getClass().getName() + '@' + Integer.toHexString(hashCode())
```



7. What do we mean by “casting”, “autoboxing”, and “wrapper class”. Compare and
contrast these terms.


_Casting_
+ tells compiler that an Object is another type, therefore gaining access to additional methods.

```java
Object o = "str";
String str = (String) o;
```


_Autoboxing_
+ conversion that compiler makes between primitives and corresponding wrapper class
  + `int` to `Integer`
  + `double` to `Double`
+ conversion the other way is called unboxing

```java
List<Integer> li = new ArrayList<>();
for (int i = 1; i < 50; i += 2)
    li.add(i);
```


8. Give four examples of subclasses of Collection in Java. Describe a different circumstance
for each in which you would require the features of that particular type
of collection. For example, how when would you use an ArrayList? How are collections
similar to arrays? How are they different? Is it possible to define a non-generic
subclass of Collection? Why or why not?


Array
+ A list of fixed number of items of same type
+ `int[] foo = new int[10]`

Collection Interface

ArrayList
+ Resizable-array implementation of the List interface.
+ fast access by index. slow insert/delete at head (start/end)
+ useful when want to store a bunch of things and iterate through them later

LinkedList
+ provides linked-list data structure
+ slow access by index but efficient insert/delete at head (start/end)

HashSet
+ extends `AbstractSet` and implements `Set` interface. Creates a collection that uses a hash table for storage
+ allows no duplicate

HashMap
+ Maps key to value and unordered.

Queue
+ FIFO

Stack
+ FILO



9. Write a main method that creates three instances of the generic class from Test \#2,
Question 3. The first instance should use Strings as the generic type, the second
should Integers, and the third should use a type that you create yourself.


```java
FavThree<Integer> myFav = new FavThree<>(new String("a"), new String("b"), new String("c"));
FavThree<Integer> myFav = new FavThree<>(new Integer(10), new Integer(20), new Integer(30));
FavThree<Box> myFav = new FavThree<>(new Box("a"), new Box("b"), new Box("c"));
```

Generics enable types (classes and interfaces) to be parameters when defining classes, interfaces and methods. type parameters provide a way for you to re-use the same code with different inputs.
+ stronger type check at compile time
+ elimination the need of cast
+ `List<E>` where `E` is an arbitrary data type

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

10.  How did we use instances of each of the following classes with regards to the
Person/Student/StudentManager examples: `Logger`, `Handler`, `Scanner`, `FileInputStream`,
`BufferedInputStream`, `ObjectInputStream`, `FileOutputStream`, `BufferedOutputStream`,
`ObjectOutputStream`. What is Serializable and how did we use it to store information
about instances of class Student?


Scanner
+ A Scanner breaks its input into tokens using a delimiter pattern, which by default matches whitespace.

```java
Scanner scanner = new Scanner(new FileInputStream(filePath));
String[] record;
Student student;

while(scanner.hasNextLine()) {
    record = scanner.nextLine().split(",");
    student = new Student(record[0].split(" "),
            record[1], record[2], record[3]);
    students.put(student.getID(), student);
}
scanner.close();
```


```java
InputStream file = new FileInputStream(path);
// Buffering reads ahead so it's faster.
InputStream buffer = new BufferedInputStream(file);
// We're reading Java objects, so we need this kind of input stream.
ObjectInput input = new ObjectInputStream(buffer);

// We can finally read the map!
map = (HashMap<String, String>) input.readObject();
input.close();
```

```java
OutputStream file = new FileOutputStream(path);
OutputStream buffer = new BufferedOutputStream(file);
ObjectOutput output = new ObjectOutputStream(buffer);
output.writeObject(map);
output.close();
```


```java
private static final Logger logger = Logger.getLogger(StudentManager.class.getName());
private static final Handler consoleHandler = new ConsoleHandler(); // receives messsage from logger and transfer to console
logger.addHandler(consoleHandler);
// Logger has different levels
```

+ Serialization is the process of saving an object's state to a sequence of bytes;
+ Deserialization is the process of rebuilding those bytes into a live object
+ Classes `ObjectInputStream` and `ObjectOutputStream` are high-level streams that contain the methods for serializing and deserializing an object.




11. Create a `JFrame` that displays the information from a `StudentManager` so that each
student has a checkbox. When the checkbox is checked, the student’s information is
sent, using the println() method, to the screen.

```java
JFrame window = new JFrame("window");



```





12. `DemoCheckedAndUnchecked.java`, `UnexpectedNegativeException.java`, and
`UnexpectedNegativeException.java`.


1. The file compiles but does not run to completion.
  + this happens with `RuntimeException`; Checked exception can terminate program if you `throws` all the way to main.
  + so just use throw a `RuntimeException` and not `catch` it or throws checked Exception all the way up.
2. The file compiles and runs, even though one of the methods throws an exception.
  + never terminates if exceptions are `catch` properly
  + so just handles `Exception` whether checked or unchecked, then program will not terminate
3. The file compiles and runs, but prints the call stack trace to the screen twice, at different parts of the program. In other words, the two traces should describe
different points in the code.
4. The file compiles and runs, even though an exception is thrown from inside a catch block.
  + the first exception sends runtime to catch block
  + inside catch block exists a nested try/catch block that raise errors and handles them
  + so file runs without termination because all errors are handled properly
5. The file compiles but does not run. However, between the moment when the last exception is thrown and the end of execution, the message “This is a message.” appears on the screen.
  + achievced with `RuntimeException` and a `System.out.prinln()` in the `finally` block


13. We discussed the following design patterns in class: `Observer`, `Singleton`, `Iterator`, and
`Strategy`. When would you want to use each pattern? Describe a situation where
the `Observer` pattern would be useful. Do that again for each of the other patterns.
Describe an alternative solution to the `Observer` pattern, the `Iterator` pattern, and
to the `Strategy` pattern.


`Observer`
+ when there is one-to-many relationship and change to the subject triggers update from observers.
+ reactive programming, where when oriented around data flow and change propagation


`Singleton`
+ restrict object creation; limit to only one instance   
+ logging class or database access service


`Iterator`
+ This pattern is used to get a way to access the elements of a collection object in sequential manner without any need to know its underlying representation.

`Strategy`
+ Enables an algorithm's behaviour to be selected at runtime.



14. Modify the file “Inherit.java” from the website under the Week 7 Readings to
demonstrate the meaning of the table on the first page of “Practice Quiz 2” under
the Week 6 Readings. In other words, create modified versions of the file to check
when Java shadows and overshadows for static/instance methods and static/instance
variables



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



15.  Try the questions in the “regex practice” file.


16. What is a floating point variable? What examples of floating point issues have we seen?

floating point approximates real numbers so as to balance trade-off between range and accuracy

problems
+ rounding
  + proper rounding used
+ arithmetic operations
  + add smaller numbers first
  + add similar numbers first; sort them
+ significant digit selection (7 appropriate for 32bit float)


18. Pretend that you are trying to explain JUnit to someone who knows nothing about it. What is an assertion? What are the different types of assertions and how do they
work? What does it mean to pass a test? What is the difference between a fail and an error? What is a “unit test”? What are the three steps to running a test? How
do you select test cases? Give examples to illustrate your explanations.

Assertion: verification that some condition is true

Test fails if assertions fails; test report error if unexpected exceptions arises.


unit test tests individual subcomponents of source code; here unit refers to methods.
