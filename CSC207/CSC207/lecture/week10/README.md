
__Singleton [article](http://www.javaworld.com/article/2073352/core-java/simply-singleton.html?page=2)__


__Observer Design Pattern__

[wiki](https://en.wikipedia.org/wiki/Observer_pattern)  
+ The observer pattern is a software design pattern in which an object, called the `subject`, maintains a list of its dependents, called `observers`, and notifies them automatically of any state changes, usually by calling one of their methods.
+ Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
+ Encapsulate the core (or common or engine) components in a Subject abstraction, and the variable (or optional or user interface) components in an Observer hierarchy.
+ The "View" part of Model-View-Controller.


Define an object that is the "keeper" of the data model and/or business logic (the `Subject`). Delegate all "view" functionality to decoupled and distinct `Observer` objects. `Observers` register themselves with the `Subject` as they are created. _Whenever the Subject changes, it broadcasts to all registered Observers that it has changed, and each Observer queries the Subject for that subset of the Subject's state that it is responsible for monitoring._

[observer pattern](assets/README-fe81e.png)

[tutorialpoints](https://www.tutorialspoint.com/design_pattern/observer_pattern.htm)
+ Observer pattern is used when there is one-to-many relationship between objects such as if one object is modified, its depenedent objects are to be notified automatically.
+ have some simple sample code...

[tutorialpoints slide](assets/README-5e09d.png)




__Runnable__

+ The Runnable interface should be implemented by any class whose instances are intended to be executed by a thread. The class must define a method of no arguments called run.


```java
public class MyRunnableTask implements Runnable {
     public void run() {
         // do stuff here
     }
}

// and use it
Thread t = new Thread(new MyRunnableTask());
t.start();
```


__[`Observer`](https://docs.oracle.com/javase/7/docs/api/java/util/Observer.html)__  
+ A class can implement the `Observer` interface when it wants to be informed of changes in `observable` objects.
+  An application calls an `Observable` object's `notifyObservers` method to have all the object's observers notified of the change.

```java
update(Observable o, Object arg)
//This method is called whenever the observed object is changed.
```



__[`Observable`](https://docs.oracle.com/javase/7/docs/api/java/util/Observable.html)__
+ This class represents an observable object, or "data" in the model-view paradigm. It can be subclassed to represent an object that the application wants to have observed.
+ An observable object can have one or more observers. An observer may be any object that implements interface `Observer`.
+  After an observable instance changes, an application calling the `Observable`'s `notifyObservers` method causes all of its observers to be notified of the change by a call to their `update` method.



__[`Singleton`](https://www.tutorialspoint.com/java/java_using_singleton.htm)__

+ The Singleton's purpose is to _control object creation_, limiting the number of objects to only one. Since there is only one Singleton instance, any instance fields of a Singleton will occur only once per class, just like static fields. Singletons often control access to resources, such as database connections or sockets.
+ The easiest implementation consists of a `private constructor` and a field to hold its result, and a static accessor method with a name like `getInstance()`.

```java
// File Name: Singleton.java
public class Singleton {

   private static Singleton singleton = new Singleton( );

   /* A private Constructor prevents any other
    * class from instantiating.
    */
   private Singleton() { }

   /* Static 'instance' method */
   public static Singleton getInstance( ) {
      return singleton;
   }

   /* Other methods protected by singleton-ness */
   protected static void demoMethod( ) {
      System.out.println("demoMethod for singleton");
   }
}
```

```java
// File Name: SingletonDemo.java
public class SingletonDemo {

   public static void main(String[] args) {
      Singleton tmp = Singleton.getInstance( );
      tmp.demoMethod( );      // prints: demoMethod for singleton
   }
}
```


Another example...

```java
public class ClassicSingleton {

   private static ClassicSingleton instance = null;
   private ClassicSingleton() {
      // Exists only to defeat instantiation.
   }

   public static ClassicSingleton getInstance() {
      if(instance == null) {
         instance = new ClassicSingleton();
      }
      return instance;
   }
}
```





__[`Iterator`]()__
+ This pattern is used to get a way to access the elements of a collection object in sequential manner without any need to know its underlying representation.


```java
public interface Iterator {
   public boolean hasNext();
   public Object next();
}
```

```java
public interface Container {
   public Iterator getIterator();
}
```


```java
public class NameRepository implements Container {
   public String names[] = {"Robert" , "John" ,"Julie" , "Lora"};

   @Override
   public Iterator getIterator() {
      return new NameIterator();
   }

   private class NameIterator implements Iterator {

      int index;

      @Override
      public boolean hasNext() {

         if(index < names.length){
            return true;
         }
         return false;
      }

      @Override
      public Object next() {

         if(this.hasNext()){
            return names[index++];
         }
         return null;
      }		
   }
}
```


```java
public class IteratorPatternDemo {

   public static void main(String[] args) {
      NameRepository namesRepository = new NameRepository();

      for(Iterator iter = namesRepository.getIterator(); iter.hasNext();){
         String name = (String)iter.next();
         System.out.println("Name : " + name);
      } 	
   }
}

// Name : Robert
// Name : John
// Name : Julie
// Name : Lora
```


__[`Iterable`](https://docs.oracle.com/javase/7/docs/api/java/lang/Iterable.html)__
+ Implementing this interface allows an object to be the target of the `foreach` statement.
+ implements `Iterator<T> iterator()` method which returns an `Iterator`

__[`Iterator`](https://docs.oracle.com/javase/7/docs/api/java/util/Iterator.html)__
+ An iterator over a collection.  
+ `hasNext()`, `next()`, `remove()`


__[Nested Class](https://docs.oracle.com/javase/tutorial/java/javaOO/nested.html)__
+ It is a way of logically grouping classes that are only used in one place
+ It increases encapsulation
+ It can lead to more readable and maintainable code





__Strategy Pattern__
+ Enables an algorithm's behaviour to be selected at runtime.
+ what it does
  + defines a family of algorithms
  + encapsulates each algorithm
  + makes the algorithms interchangeable within that family
+ For instance, a class that performs validation on incoming data may use a strategy pattern to select a validation algorithm based on the type of data, the source of the data, user choice, or other discriminating factors. These factors are not known for each case until run-time, and may require radically different validation to be performed.



__[`Comparable<T>`](https://docs.oracle.com/javase/7/docs/api/java/lang/Comparable.html)__
+ This interface imposes a total ordering on the objects of each class that implements it. This ordering is referred to as the class's natural ordering, and the class's `compareTo` method is referred to as its natural comparison method.
+ Lists (and arrays) of objects that implement this interface can be sorted automatically by `Collections.sort`
+ The _natural ordering for_ a class C is said to be consistent with `equals` if and only if `e1.compareTo(e2) == 0` has the same boolean value as `e1.equals(e2)` for every e1 and e2 of class C. Note that null is not an instance of any class, and e.compareTo(null) should throw a `NullPointerException` even though e.equals(null) returns false.


```java
int compareTo(T o)
```

Compares this object with the specified object for order. Returns a negative integer, zero, or a positive integer as this object is less than, equal to, or greater than the specified object.
+ ensures `(x.compareTo(y)) == -sgn(y.compareTo(x))`
+ ensures transitivity `(x.compareTo(y)>0 && y.compareTo(z)>0)` implies` x.compareTo(z)>0`.
+ ensures `x.compareTo(y)==0` implies that `sgn(x.compareTo(z)) == sgn(y.compareTo(z))`, for all z.
+ It is strongly recommended, but not strictly required that `(x.compareTo(y)==0) == (x.equals(y))`


```java
InsertionSorter<T extends Comparable<T>>
```

This means that the type parameter must support comparison with other instances of its own type, via the `Comparable` interface.



__RegExp__
