


__Testing__
+ unit test has method as its subject


Some of the important assertions
+ `assertEqual(expected, actual)`
+ `assertTrue(booleanExpression)`
+ `assertNotNull(Object)`
+ `fail` 

`setUp()` and `tearDown()`
+ ran for each method calls  

`@Before`
+ run before every `@Test`

`@After`
+ run after every `@Test`

`@BeforeClass`
+ run once before all the tests

`@AfterClass`
 + run once after all the tests


`@Test`
The Test annotation tells JUnit that the public void method to which it is attached can be run as a test case. The Test annotation supports two optional parameters. The first, expected, declares that a test method should throw an exception. If it doesn't throw an exception or if it throws a different exception than the one declared, the test fails. For example, the following test succeeds:

```java
@Test(expected=IndexOutOfBoundsException.class)
public void outOfBounds() {
   new ArrayList<Object>().get(1);
}
```


```java
@Test
public void testIndexOutOfBoundsException() {
    ArrayList emptyList = new ArrayList();
    try {
        Object o = emptyList.get(0);
        AssertFail(“IndexOutOfBoundsException not thrown: 0.”);  // assert fail here...
     } catch (IndexOutOfBoundsException e) {
     }
}
```

__Logging__

+ A `Logger` object is used to log messages for a specific system or application component.

+ Logger objects may be obtained by calls on one of the `getLogger` factory methods.

+ Logging messages will be forwarded to registered `Handler` objects, which can forward the messages to a variety of destinations, including consoles, files, OS logs, etc.

+ Each logger has a `Level` associated with it. This reflects a minimum Level that this logger cares about.


__Handler__

+ A Handler object takes log messages from a Logger and exports them. It might for example, write them to a console or write them to a file, or send them to a network logging service, or forward them to an OS log, or whatever.

+ A Handler can be disabled by doing a `setLevel(Level.OFF)` and can be re-enabled by doing a setLevel with an appropriate level.




__HashMap<K, V>__  

+ Hash table based implementation of the `Map` interface.




__File__

+ An abstract representation of file and directory pathnames.

```java
new File(pathname) // constructor
```
