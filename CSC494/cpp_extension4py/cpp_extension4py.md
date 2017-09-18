


#### Python/C API [reference](https://docs.python.org/2/c-api/index.html)


##### Intro 

+ `Python.h`
    + in `~/anaconda/include/python2.7`
+ _Object_
    + `PyObject *`
        + a pointer to an opaque data type representing arbitrary objects 
        + points to object living in heap, 
            + excluding `PyTypeObject` which is static
    + _consists of_
        + _type_    
            + can be checked with `PyList_CHeck(foo)`
        + _reference count_ 
+ _reference count_    
    + number of reference to an `PyObject` by
        + another object 
        + a global C variable 
        + local variable in some C function 
    + _functionality_ 
        + object deallocated when reference count -> 0
        + recursively decrement owning objects' reference counts
        + JUST like a `shared_ptr`... but cant do circular reference...
    + _management explicitly only_ 
        + `Py_INCREF()` 
        + `Py_DECREF()`
            + decrement member object's reference count
            + might call _deallocator_, a function pointer in object's type structure 
        + so idea is manually increment/decrement reference count only if dont want to have local objects deallocated when go out of scope
    + _behavior_ 
        + _local variable_ 
            + increment on creation, decrement when goes out of scope
        + _argument passed to extension module function_
            + guaranteed to persists, no need to manage reference count 
    + _convention_ 
        + dangerous to extract object from list...
        + use generic operations `PyObject_`, `PyNumber_`, `PySequence_`, `PyMapping`
            + always increment reference count of object they return 
            + leaves caller to call `Py_DEREF()`
    + _ownership of references_ 
        + owning reference means responsible for calling `Py_DECREF` on it.
        + objects are always shared 
    + _funtion passes reference to caller_ 
        + if ownership transderred, caller is said to _receive_ `new` reference 
        + if no ownership transferred, caller said to _borrow_ reference, (i.e. non-owning, just like extension argument)
    + _calling function passes in a reference to an object_
        + function _steals reference_ to object
            + function now owns the reference
            + few function steal reference
                + except `PyList_SetItem()`, `PyTuple_SetItem()`, which steal reference to item 
                + _idiom_: populate tuple/list with newly created objects   
                ```cpp
                // `(1, 2, 'three')` 
                PyObject *t;
                t = PyTuple_New(3);
                PyTuple_SetItem(t, 0, PyInt_FromLong(1L));
                PyTuple_SetItem(t, 1, PyInt_FromLong(2L));
                PyTuple_SetItem(t, 2, PyString_FromString("three"));
                ```
                + `PyInt_FromLong()` returns a new reference which is immediately stolen by `PyTuple_SetItem()`
        + _or not_ 
            + `PySequence_SetItem()` or `PyObject_SetItem()`
    + `Py_BuildValue()`
        + creates most common objects from C values, by a format string
        ```cpp 
        PyObject* tuple, *list;

        tuple = Py_BuildValue("(iis)", 1, 2, "three");
        list = Py_BuildValue("[iis]", 1, 2, "three");
        ```
    + `PyObject_SetItem()`
        + for borrowing references only,
        ```cpp 
        // sets all items of a list to a given item 
        int set_all(PyObject* target, PyObject *item)
        {
            int i, n;

            n = PyObject_Length(target);
            if(n < 0)
                return -1;
            for(int i; i < n; i++)
            {
                PyObject* index = PyInt_FromLong(i);
                if(!index)
                    return -1;
                if(PyObject_SetItem(target, index, item) < 0){
                    PY_DECREF(index);
                    return -1;
                }
                PY_DECREF(index);
            }
            return 0;
        }
        ```
    + _function argument_
        + does not change ownership responsibilities for reference (most of the time)
        + so using `PyObject_SetItem()` on `item` only borrows so no need to increment reference count
    + _function return values_ 
        + many function return a reference give caller ownership of reference
        + like `PyObject_GetItem()`, `PySequence_GetItem()` returns a new reference 
    + _ownership depends on function not return type_ 
        + `PyList_GetItem()` returns borrowed reference
        + `PySequence_GetItem()` returns new reference, ownership transferred
        ```cpp 
        // PyList_GetItem()
        long sum_list(PyObject* list)
        {
            int i, n;
            long total = 0;
            PyObject* item;

            n = PyList_Size(list);
            if(n < 0)
                return -1   // not a list
            for(i = 0; i < n; i++){
                item = PyList_GetItem(list, i);     // cannot fail
                if(!PyInt_Check(item)) continue;    // skip non-int
                total += PyInt_AsLong(item);
            }
            return total;
        }
        ```
        ```cpp 
        // PySequence_GetItem()
        long
        sum_sequence(PyObject *sequence)
        {
            int i, n;
            long total = 0;
            PyObject *item;
            n = PySequence_Length(sequence);
            if (n < 0)
                return -1; 
            for (i = 0; i < n; i++) {
                item = PySequence_GetItem(sequence, i);
                if (item == NULL)       // not a sequence
                    return -1; 
                if (PyInt_Check(item))
                    total += PyInt_AsLong(item);
                Py_DECREF(item); // Discard reference ownership 
            }
            return total;
        }
        ```
        + `Py_ssize_t`
            + usually `typedef` to `ssize_t` for signed size 
+ _Types_ 
    + mostly `int`, `long`, `double`, `char*`
+ _Exceptions_ 
    + have to deal with exceptions explicitly in C
    + _situation_ 
        + function encounter error 
        + sets exception 
        + discards any object references it owns
        + returns error indicator `NULL` or `-1`
    + _exception state_ 
        + maintained in per-thread storage
        + _occurred_ 
            + `PyErr_Occurred()` check 
                + returns borrowed reference to exception type object, `NULL` otherwise
            + `PyErr_ExceptionMatches()`
                + verifies which exception
        + _or not_
    + _set state_ 
        + `PyErr_SetString()`
    + _clear state_ 
        + `PyErr_Clear()`
    + _maps to Python version in interpreter_
        + `sys.exc_type`
        + `sys.exc_value`
        + `sys.exc_traceback`
    ```py
    def incr_item(dict, key):
        try:
            item  = dict[key]
        except keyError:
            item = 0
        dict[key] = item + 1;
    ```
    ```cpp 
    int incr_item(PyObject *dict, PyObject *key)
    {
        PyObject *item =NULL, *const_one = NULL, *incremented_item = NULL;
        int rv = -1;        // return value

        item = PyObject_GetItem(dict, key);
        if(item == NULL){
            // key error only 
            if(!PyErr_ExceptionMatches(PyExc_KeyError))
                goto error;
            // clear error and use zero 
            item = PyInt_FromLong(0L);
            if(item == NULL)
                goto error;
        }

        const_one = PyInt_FromLong(1L);
        if(const_one == NULL)
            goto error;
        incremented_item = PyNumber_Add(item, const_one);
        if(incremented_item == NULL)
            goto error;
        rv = 0;

    error:
        // clean up code 
        Py_XDECREF(item);
        Py_XDECREF(const_one);
        Py_XDECREF(incremented_item);
        return rv;
    }
    ```
    + _note_ 
        + use `PyError_ExceptionMatches()` to handle  excecption 
        + use `PyXDECREF()` to dispose owned reference that maybe `NULL`
+ _Embedding python_ 
    + `Py_Initialize()`
        + initialize table of loaded modules 
        + creates modules like `__builtin__`, `__main__`, `sys`, `exceptions`
        + calculate module search path 
            + try to look for directory `lib/pythonX.Y` relative to parent directory where `python` executable is found
            + `/usr/local/bin/python` to `/usr/local/lib/pythonX.Y`

##### Very high level layer 

+ `Py_Main(int argc, char **argv)`

##### Reference Counting 

+ `void Py_INCREF(PyObject *o)`
+ `void Py_XINCREF(PyObject *o)`
+ `void Py_DECREF(PyObject *o)`
+ `void Py_XDECREF(PyObject *o)`
+ `void Py_CLEAR(PyObject *o)`



##### Common Object Structure [doc](https://docs.python.org/2/c-api/structures.html#c.PyObject)

+ `PyObject`
    + all object types are extension of this type
    + a type which contains info Python needs to treat a pointer to an object as an object 
    + _contains_ 
        + _reference count_ 
        + _a pointer to corresponding type object_
    + _corresponds_ 
        + to field defined by expansion of `PyObject_HEAD` macro 
+ `PyVarObject`
    + extension of `PyObject` that adds `ob_size` field, 
    + used for objects with notion of _length_ 
    + _corresponds_ 
        + expansion of `PyObject_VAR_HEAD`
+ `PyObject_HEAD`
    + expands to declaration of fields of `PyObject` types
    ```cpp 
    Py_ssize_t ob_refcnt;
    PyTypeObject *ob_type;
    ```
+ `PyObject_VAR_HEAD`
    ```cpp 
    PyObject_HEAD
    Py_ssize_t ob_size;
    ```
+ `PyMethodDef`
    ```cpp 
    struct PyMethodDef {
        char *ml_name;
        PyCFunction ml_meth;
        int ml_flags;
        char *ml_doc;
    }
    ```


### Extending and Embedding Python interpreter [v2.7](https://docs.python.org/2/extending/index.html)


#### Extending python with C/C++ [v2.7](https://docs.python.org/2/extending/extending.html)

```py
# want
import spam
status = spam.system("ls -l")
```
```c 
#include <Python.h>

static PyObject *
spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    if (sts < 0) {
        PyErr_SetString(SpamError, "System command failed");
        return NULL;
    }
    return PyLong_FromLong(sts);
}
```
+ _note_ 
    + _translating arg list_ 
        + python: `"ls -l"` in `system("ls -l")`
        + C: `command` now holds `"ls -l"`
    + _function signature_ 
        + `PyObject* self`
            + `NULL` or pointer selected while initializing module `Py_InitModule4()`
            + points to object instance 
        + `PyObject* args`
            + a pointer to a Python tuple object containing the arguments 
            + each item corresponds to an argument in the call's arg list 
            + each item is a `PyObject`
    + `PyArg_ParseTuple` 
        + checks arg type and converts to C values
        + returns `True` if all args have right type and components have been stored in varaibles whose addresses are passed in
        + Note the function itself sets exception already...
    + `PyBuildValue(const char*format, ...)`
        + creates a new value based on format string 
        + returns the value or NULL in case of error
+ _Errors/Exceptions_ 
    + _convention for functions_
        + sets exception condition 
        + return `NULL` 
    + _storing exception_  
        + `sys.exc_type`: exception type
        + `sys.exc_value`: associated value of exception
        + `sys.exc_traceback`: stack traceback
    + _Setting_ 
        + `PyErr_SetString(PyObject* type, const cahr*msg)`
            + `type`: exception type 
                + i.e.  predefined objects `PyExc_RuntimeError`
        + `PyErr_SetFromErrno()`
            + construct value by inspecting `errno`
        + `PyErr_SetObject(PyObject* type, PyObject* value)`
    + _property_ 
        + always the function that first detecting the error calls `PyErr_*()` 
        + functions away from exception init position relays error and not setting them 
    + _clearing_ 
        + `PyErr_Clear()`
            + clears error
    + _handling `malloc`_
        + call `PyErr_NoMemory()`
    + _custom error type_ 
        + `PyErr_NewException`
            + create and return a new exception class
        + `static PyObject *SpamError;`

```c 
PyMODINIT_FUNC
initspam(void)
{
    PyObject *m;

    m = Py_InitModule("spam", SpamMethods);
    if (m == NULL)
        return;

    SpamError = PyErr_NewException("spam.error", NULL, NULL);
    Py_INCREF(SpamError);
    PyModule_AddObject(m, "error", SpamError);
}
```
+ _note_ 
    + `Py_InitModule(char *name, PyMethodDef *methods)`
        + creates a new module object based on name a table of functions
    + `Py_INCREF`
        + used to ensure `SpamError` is not dangling
    + `PyModule_AddObject(PyObject* module, const char*name, PyObject" value)`
        + add an object to module as `name`
        + steals a ref to `value`, 
        + `-1` on failure, `0` on success


+ _Return values_ 
    + `Py_BuildValue`
    + return `void`, use idiom 
        ```c 
        Py_INCREF(Py_None);
        return Py_None;
        ```
        + `Py_None` is C name for special Python object `None`
            + not a pointer! but an object


+ _Method table and initialization function_ 
    ```c
    static PyMethodDef SpamMethods[] = {
    {"system",  spam_system, METH_VARARGS,
     "Execute a shell command."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
    };  

    PyMODINIT_FUNC
    initspam(void)
    {
        (void) Py_InitModule("spam", SpamMethods);
    }
    ```
    + _note_ 
        + `METH_VARARGS`
            + flag telling interpreter the convention to be used for C functions 
            + `METH_VARARGS`, or
            + `METH_VARARGS | METH_KEYWORDS`
                + function have 3rd arg `PyObject *` for dictionary of keywords 
                + use `PyArg_ParseTupleAndKeywords()` to parse args to function
        + `initspam`
            + function called on `import spam`
        + `Py_InitModule()`
            + creates a module objects 
            + inserted in dictionary `sys.modules`s under key `spam` 
            + inserts built-in function objects into the module based on the table `PyMethodDef` array
+ _Compilation and Linkage_ 
+ _Extracting Parameters in Extension functions_ 
    + `int PyArg_ParseTuple(PyObject *arg, char *format, ...);`
        + `arg` is a tuple object 
        + `format` is a string 
        ```cpp 
        int ok;
        int i, j;
        long k, l;
        const char *s;
        int size;

        // f()
        ok = PyArg_Parse(args, "");  // no args 

        // f('hi')
        ok = PyArg_ParseTuple(args, "s", &s);

        // f(1,2,'three')
        ok = PyArg_ParseTuple(args, "lls", &k, &l, &s); // 2 long and a string

        // f((1,2), 'three')
        ok = PyArg_ParseTuple(args, "(ii)s#", &i, &j, &s, &size); // pair of ints and a string with size 
        ```
        ```cpp 
        const char *file
        const char *mode;
        int bufsize = 0;

        // f('spam')
        // f('spam' ,'w')
        // f('spam', 'w', 100);
        ok = PyArg_ParseTuple(args, "s|si", &file, &mode, &bufsize); // a string, optionally another string + int
        ```
+ _Keyword parameters for extension functions_
    + `int PyArg_ParseTupleAndKeywords(PyObject *arg, PyObject *kwdict, char *format, char *kwlist[], ...);`
        + `kwdict`: dictionary of keywords received as 3rd arg from Python runtime 
        + `kwlist` null terminated list of strings that identify the params
    ```cpp 
    static PyObject *
    keywdarg_parrot(PyObject *self, PyObject *args, PyObject *keywds)
    {
        int voltage;
        char *state = "a stiff";
        char *action = "voom";
        char *type = "Norwegian Blue";

        static char *kwlist[] = {"voltage", "state", "action", "type", NULL};

        if (!PyArg_ParseTupleAndKeywords(args, keywds, "i|sss", kwlist,
                                        &voltage, &state, &action, &type))
            return NULL;

        printf("-- This parrot wouldn't %s if you put %i Volts through it.\n",
            action, voltage);
        printf("-- Lovely plumage, the %s -- It's %s!\n", type, state);

        Py_INCREF(Py_None);

        return Py_None;
    }

    static PyMethodDef keywdarg_methods[] = {
        /* The cast of the function is necessary since PyCFunction values
        * only take two PyObject* parameters, and keywdarg_parrot() takes
        * three.
        */
        {"parrot", (PyCFunction)keywdarg_parrot, METH_VARARGS | METH_KEYWORDS,
        "Print a lovely skit to standard output."},
        {NULL, NULL, 0, NULL}   /* sentinel */
    };

    void
    initkeywdarg(void)
    {
    /* Create the module and add the functions */
    Py_InitModule("keywdarg", keywdarg_methods);
    }
    ```    
+ _Building arbitrary values_ 
    + `PyObject *PyBuildValue(char *format, ...);`
        + `...`: values 
        + returns new Python object
    ```cpp 
    Py_BuildValue("")                       //  None
    Py_BuildValue("i", 123)                 // 123
    Py_BuildValue("iii", 123, 456, 789)     // (123, 456, 789)
    Py_BuildValue("s", "hello")              // 'hello'
    Py_BuildValue("ss", "hello", "world")    // ('hello', 'world')
    Py_BuildValue("s#", "hello", 4)          // 'hell'
    Py_BuildValue("()")                      // ()
    Py_BuildValue("(i)", 123)                // (123,)
    Py_BuildValue("(ii)", 123, 456)          // (123, 456)
    Py_BuildValue("(i,i)", 123, 456)         // (123, 456)
    Py_BuildValue("[i,i]", 123, 456)         // [123, 456]
    Py_BuildValue("{s:i,s:i}",
                "abc", 123, "def", 456)     // {'abc': 123, 'def': 456}
    Py_BuildValue("((ii)(ii)) (ii)",
                1, 2, 3, 4, 5, 6)          // (((1, 2), (3, 4)), (5, 6))
    ```
+ _Reference counts_ 
    + _cycle detector_ 
        + detecs reference cycles
    + nobody owns an object, but you can own a reference to an object
    + _types_ 
        + _owner of reference_  C++ `shared_ptr`
            + responsible for calling `Py_DECREF()` when reference no longer neededs
        + _borrower of reference_  C++ `&` 
            + should not call `Py_DECREF()`
            + may lead to dangling reference...
            + calling `Py_INCREF()` makes the _borrower_ an _owner_
    + _ownership rules_ 
        + most function returning a reference to an object pass on ownership with the reference 
            + _object creating function_ 
                + `PyInt_FromLong()`, `Py_BuildValue()`
            + _function that extract objects from other objects_
                + `PyObject_GetAttrString()`
            + _extension function_ 
                + returned reference had ownership transferred to caller
        + some exception, function that returns borrowed reference instead 
            + _object that extract objects_ 
                + `PyTuple_GetItem()`, `PyList_GetItem()`, `PyDict_GetItem()`, `PyDict_GetItemString()`
            + `PyImport_AddModule()`
                + ownership stored in `sys.modules`
        + Usually, passing object reference to another function _borrows_ reference
            + use `Py_INCREF()` if want to store the value inside the function 
            + _arguments in extension function_
                + owned by Python runtime
                + guaranteed lifetime until function returns
                + no need to `Py_DECREF()`
        + some exception, function that _steals_ reference
            + `PyTuple_SetItem()`, `PyList_SetItem()`
                + function takes over ownerhsip of passed item 
+ _Thin ice_ 
    + _Problem_  
        + _using `Py_DECREF()` on an unrelated object while borrowing a reference to a list item_ 
    ```cpp 
    void
    bug(PyObject *list)
    {
        // borrows reference from list[0], non-owning
        PyObject *item = PyList_GetItem(list, 0);

        // replace list[1] with 0
        PyList_SetItem(list, 1, PyInt_FromLong(0L));

        // prints borrowed reference
        PyObject_Print(item, stdout, 0); /* BUG! */
    }
    ```
    + _note_ 
        + `list` owns reference to all items 
        + when `list[1]` replaced, the original item is disposed
        + original item is user-defined class with `__del()__` defined, will be called if reference count is 1
        + `__del()__` risks invalidating reference to `list[0]` or `item`
            + with `del list[0]`
        + `item` dangles
    ```cpp 
    void
    no_bug(PyObject *list)
    {
        PyObject *item = PyList_GetItem(list, 0);

        Py_INCREF(item);
        PyList_SetItem(list, 1, PyInt_FromLong(0L));
        PyObject_Print(item, stdout, 0);
        Py_DECREF(item);
    }
    ```
    + _solution_ 
        + temporarily increment reference counts to make `item` valid 
    + _Problem_     
        + borrowed reference is a variant invovling threads
        + may temporarily release the global lock with `Py_BEGIN_ALLOW_THREADS` `Py_END_ALLOW_THREADS`
        + i.e. IO calls
    ```cpp 
    void
    bug(PyObject *list)
    {
        PyObject *item = PyList_GetItem(list, 0);
        Py_BEGIN_ALLOW_THREADS
        // ...some blocking I/O call...
        // may invalidate item in a different thread
        Py_END_ALLOW_THREADS
        PyObject_Print(item, stdout, 0); /* BUG! */
    }
    ```
+ _NULL Pointer_ 
    + _convention_ 
        + function will dump core if pass `NULL` as arguments to function
        + function that return object references generally return `NULL` only to indicate that an exception has occurred
        + test for `NULL` at _source_ 
            + when a pointer is just received
    + _behavior_ 
        + `PyObject* args` is never `NULL`
+ _Providing C API for an extension module_ 
    + API defined in an extension module, that can be used in other extension modules
    + like `numpy`
    + _getaway_ 
        + all symbols in extension modules should be `static`, except for initialization function
    + _Capsules_ 
        + mechanism to pass C-level information (pointer) from one extension module to another one
        + a data type which stores a pointer `void *`
        + can be passed around like Python objects, but can be created and accessed via C API
        + can be imported to retrieve content of capsules
    + `PyCapsule_New(void*ptr, const char* name, dtor)`
        + createa a `PyCapsule` encapsulating a pointer
        + `PyCasule` is a subtype of `PyObject`, represent an opaque data value
    + `modulename.attributename`
    + `PyCapsule_Import()`
    ```cpp 
    // client module
    static int PySpam_System(const cahr* command)
    {
        return system(command);
    }

    static PyObject* 
    spam_system(PyObject *self, PyObject *args)
    {
        cosnt char *command;
        int sts;

        if(!PyArg_ParseTuple(args, "s", &command))
            return NULL;
        sts = PySam_System(command);
        return Py_BuildValue("i", sts);
    }
    ```
    ```cpp 
    // exporting module (not client)
    #include "Python.h"
    
    #define SPAM_MODULE
    #include "spammodule.h"
    ```
    + _note_ 
        + it is being included in the exporting module, not a client module
    ```cpp 
    PyMODINIT_FUNC
    initspam(void)
    {
        PyObject *m;
        static void *PySpam_API[PySpam_API_pointers];
        PyObject *c_api_object;

        m = Py_InitModule("spam", SpamMethods);
        if (m == NULL)
            return;

        /* Initialize the C API pointer array */
        PySpam_API[PySpam_System_NUM] = (void *)PySpam_System;

        /* Create a Capsule containing the API pointer array's address */
        c_api_object = PyCapsule_New((void *)PySpam_API, "spam._C_API", NULL);

        if (c_api_object != NULL)
            PyModule_AddObject(m, "_C_API", c_api_object);
    }
    ```
    
#### Defining New Types [doc](https://docs.python.org/2/extending/newtypes.html)


+ _basics_ 
    + runtime sees all python object as `PyObject*`
    + `PyObject` 
        + refcount 
        + pointer to object's _type object_
    + _type object_ 
        + determines which C function gets called when
            + so called _type methods_ 
            + ie. an attribute gets looked up 
    ```cpp 
    #include <Python.h>

    typedef struct {
        PyObject_HEAD
        /* Type-specific fields go here. */
    } noddy_NoddyObject;

    static PyTypeObject noddy_NoddyType = {
        // type object is still a python object, so has its 
        // own type object, initialize refcount, pointer with this macro
        PyVarObject_HEAD_INIT(NULL, 0)         
        "noddy.Noddy",             /* tp_name */
        sizeof(noddy_NoddyObject), /* tp_basicsize */
        0,                         /* tp_itemsize */
        0,                         /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_compare */
        0,                         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,        /* tp_flags */
        "Noddy objects",           /* tp_doc */
    };

    static PyMethodDef noddy_methods[] = {
        {NULL}  /* Sentinel */
    };

    #ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
    #define PyMODINIT_FUNC void
    #endif
    PyMODINIT_FUNC
    initnoddy(void) 
    {
        PyObject* m;

        noddy_NoddyType.tp_new = PyType_GenericNew;
        if (PyType_Ready(&noddy_NoddyType) < 0)
            return;

        m = Py_InitModule3("noddy", noddy_methods,
                        "Example module that creates an extension type.");

        Py_INCREF(&noddy_NoddyType);
        PyModule_AddObject(m, "Noddy", (PyObject *)&noddy_NoddyType);
    }
    ```
    + _note_ 
        + `noddy_NoddyObject` 
            + a python object
            + what `PyObject *` will point to 
        + `PyTypeObject noddy_NoddyType`
            + type object
        + `PyVarObject_HEAD_INIT(&PyType_Type, 0)`
            + instead of `PyVarObject_HEAD_INIT(NULL, 0)`
            + `PyObject* PyType_Type` is the type object for type object
        + `PyType_Ready(PyTypeObject *type)`
            + fnialize type object, called on all types to finish initialization
            + responsible for adding inherited slots from type's base class 
        + `tp_itemsize` 
            + has to do with lists and strings
        + `Py_TPFLAGS_DEFAULT`
            + skipping a number of type methods dont provide
        + `tp_new`
            + function for creation
        + `PyModule_AddObject()`
            + adds the type to module dictionarys
            ```py 
            import noddy 
            mynoddy = noddy.Noddy()
            ```
+ _Adding data and methods_ 
    + _destructor_ 
        + `Noddy_dealloc(Noddy* self)` 
            + assigned to `tp_dealloc`
            + decrements reference count of 2 python attributes
        + `tp_free` is a member of object type `NoddyType`, used to free the object type's memory
    + _constructor_
        + `Noddy_new(PyTypeObject *type, PyObject *args, PyObject *kwds)`
            + assigned to `tp_new`, as `__new__()`
            + for allocating memory for a new instance 
                + may choose to default initialize members
            + if dont care about initial value, use `PyType_GenericNew()`
                + which initialize all member variable to `NULL`
            + `type` argument 
                + the type being instantiated 
            + `args`
                + arguments passed when type is called
                + usually ignored
            + _return_
                + new instance of python object created
            + _inheritance problems_ 
                + if `type` supports subclassing, type passed maynot be the type being defined
                + `self = (Noddy *)type->tp_alloc(type, 0)` 
                    + here, method calls `tp_alloc` for allocating memory 
                    + dont fill `tp_alloc` ourselves
                    + insatead rely on `PyType_Ready()` to fill for us by inheriting it from base class
                + might have to use `type->tp_case->tp_new` for a co-operative `tp_new`
    + _initialization_ 
        + `Noddy_init(Noddy *self, PyObject *args, PyObject *kwds)`
            + assigned to `tp_init`, as `__init__()`
            + for initializing values, potentially from `args`, to members of an allocated instance `self`
            + _does not guaranteed to be called_
                + unpicking does not call 
                + can be overriden
            + _can be called multiple times_ 
                ```cpp
                if(first){
                    Py_XDCREF(self->first); // allocated member
                    Py_INCREF(first);       // extracted from args
                    self->first = first;
                }
                ```
                + _note_ 
                    + maybe risky since anyonce can call `__init__()`
                    + does not restrict type of `first`
                        + maybe any type
                        + could have a destructor that causes code to access `first` member
                + _solution_ 
                    + always reassign member before decrementing reference counts
                ```cpp 
                if(first){
                    tmp = self->first;      
                    Py_INCREF(first);
                    self->first = first;
                    Py_XDECREF(tmp);
                }
                ```
    + _exposing member as attributes_ 
        + _define member definitions_
            + define `PyMemberDef` array 
                + `static PyMemberDef Noddy_members[] = { ... };`
            + put in `tp_members`
            + has member name, type, offset, access flags, doc string
    + _define member function_ 
        + `static PyObject *Noddy_name(Noddy* self)`
            + return concat of first and last name of `Noddy`
            + _equivalence_
            ```py
            def name(self):
                return "%s %s" % (self.first, self.last)
            ```
            + _note_    
                + always check for `NULL` for member i.e. `self->first`, `self->second`
        + register to `PyMethodDef` array 
            ```cpp 
            static PyMethodDef Noddy_methods[] = {
                {"name", (PyCFunction)Noddy_name, METH_NOARGS, "return name, combining first and last name"},
                {NULL}
            };
            ```
        + _assigns to `tp_methods`_
    + _allow subclassing (as baseclass)_
        + `Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/`
+ _Finer control over data attributes_ 
    + _goal_ 
        + ensure `self->first` and `self->last` does not set to non-string values or be deleted
    + _custom getter/setter_ 
        ```cpp 
        Noddy_getfirst(Noddy *self, void *closure)
        {
            Py_INCREF(self->first);
            return self->first;
        }

        static int
        Noddy_setfirst(Noddy *self, PyObject *value, void *closure)
        {
        if (value == NULL) {
            PyErr_SetString(PyExc_TypeError, "Cannot delete the first attribute");
            return -1;
        }

        if (! PyString_Check(value)) {
            PyErr_SetString(PyExc_TypeError,
                            "The first attribute value must be a string");
            return -1;
        }

        Py_DECREF(self->first);     // prev obj refcnt -> 0 
        Py_INCREF(value);           // value refcnt -> 2 (later -> 1)
        self->first = value;        // does not change refcnt

        return 0;
        }
        ```
        + _note_ 
            + use `Py_INCREF()` on member to the return object such that ownership is either transferred or shared 
            + use `Py_DECREF()` on previous object when trying to set a new value to a member 
    + `PyGetSetDef`
        ```cpp 
        static PyGetSetDef Noddy_getseters[] = {
            {"first" , (getter)Noddy_getfirst, (setter)NOddy_setfirst, "firstname" , NULL}, 
            ...
        }
        ```
    + _assign `Noddy_getseters` to `tp_getset`_ 
    + _restrict args passed_ 
        + change format faorm `"|OOi"` to `"|SSi"`
+ _Cyclic garbage collection_ 
    + _cyclic GC_
        + identify even when reference count are not zero,
        ```py
        l = []
        l.append(l)
        del l
        ```
        + _note_    
            + a list containing itself 
            + upon `del`, reference count never zero
            + but GC takes care of cyclic referenceand free properly
    + _goal_ 
        + allow member `Noddy->first` to be arbitrary type, so maybe `Noddy` itself
        + implies possibility of a cycle
        ```py 
        import noddy2
        n = noddy2.Nodd()
        l = [n]
        n.first = l
        ```
    + _adding support for cyclic-GC_
        ```cpp 
        static int
        Noddy_traverse(Noddy *self, visitproc visit, void *arg)
        {
            int vret;

            if (self->first) {
                vret = visit(self->first, arg);
                if (vret != 0)
                    return vret;
            }
            if (self->last) {
                vret = visit(self->last, arg);
                if (vret != 0)
                    return vret;
            }

            return 0;
        }
        ```
        + _note_ 
            + traversal methods provides access to subobjects that could participate in cycles
            + how `visit` applies to `self`
            + _macro for convenience_
            ```cpp 
            Py_VISIT(self->first);
            Py_VISIT(self->last);
            ```
        ```cpp 
        static int
        Noddy_clear(Noddy *self)
        {
            PyObject *tmp;

            tmp = self->first;
            self->first = NULL;
            Py_XDECREF(tmp);

            tmp = self->last;
            self->last = NULL;
            Py_XDECREF(tmp);

            return 0;
        }
        ```
        + _note_    
            + used to clear subobjects that can participate in cycles
            + use `tmp` so that we can set each member to `NULL` before decrementing reference count
                + prevent situation where destructor (with undefined behavior) called when reference count reach 0 
            + _macro_ 
            ```cpp 
            Py_CLEAR(self->first);
            Py_CLEAR(self->last);
            ```
        ```cpp 
        static void
        Noddy_dealloc(Noddy* self)
        {
            PyObject_GC_UnTrack(self);
            Noddy_clear(self);
            Py_TYPE(self)->tp_free((PyObject*)self);
        }
        ```
        + _note_ 
            + _case_ 
                + `dealloc` may call arbitrary functions via `__del__` methods 
                + GC can be triggered inside the function
                + but GC assumes reference count not zero
                + need to untrack object from GC by calling `PyObject_GC_UnTrack()` before clearing members
        + _add `tp_flags`_
            + `Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */`
+ _Subclassing other types_ 
    + _fact_ 
        + possible to create new extension type that are derived from existing types 
        + easiest to inherit from built in types
        + can be difficult to share `PyTypeObject` between extension modules
    + `Shoddy` inherits built-in type `list`
    ```cpp 
    typedef struct {
        PyListObject list;
        int state;
    } Shoddy;
    ```
    + _note_ 
        + _derived type object_ must have base type's object structure be its first struct member
            + `PyObject_HEAD()` is included inside of `PyListObject`
        + _implication_ 
            + `Shoddy *` can be safely cast to `PyObject *` and `PyListObject *`
    ```cpp 
    static int
    Shoddy_init(Shoddy *self, PyObject *args, PyObject *kwds)
    {
        if (PyList_Type.tp_init((PyObject *)self, args, kwds) < 0)
        return -1;
        self->state = 0;
        return 0;
    }

    PyMODINIT_FUNC
    initshoddy(void)
    {
        PyObject *m;

        ShoddyType.tp_base = &PyList_Type;
        if (PyType_Ready(&ShoddyType) < 0)
            return;

        m = Py_InitModule3("shoddy", NULL, "Shoddy module");
        if (m == NULL)
            return;

        Py_INCREF(&ShoddyType);
        PyModule_AddObject(m, "Shoddy", (PyObject *) &ShoddyType);
    }
    ```
    + _initialization_ 
        + `new` methods should not actually create memory for object with `tp_alloc`, that will be handled by base class when calliing its `tp_new`
        + `tp_base()`
            + should not fill slot for `tp_base()` for cross-platform issues
            + must be done later in module `init()` function with 
                + `ShoddyType.tp_base = &PyList_Type;`
        + `PyTypeReady()` 
            + responsible for adding inherited slots from a type's base class
+ _Type methods_ 
    ```cpp 
    typedef struct _typeobject {
    PyObject_VAR_HEAD
    char *tp_name; /* For printing, in format "<module>.<name>" */
    int tp_basicsize, tp_itemsize; /* For allocation */

    /* Methods to implement standard operations */

    destructor tp_dealloc;
    printfunc tp_print;
    getattrfunc tp_getattr;
    setattrfunc tp_setattr;
    cmpfunc tp_compare;
    reprfunc tp_repr;

    /* Method suites for standard classes */

    PyNumberMethods *tp_as_number;
    PySequenceMethods *tp_as_sequence;
    PyMappingMethods *tp_as_mapping;

    /* More standard operations (here for binary compatibility) */

    hashfunc tp_hash;
    ternaryfunc tp_call;
    reprfunc tp_str;
    getattrofunc tp_getattro;
    setattrofunc tp_setattro;

    /* Functions to access object as input/output buffer */
    PyBufferProcs *tp_as_buffer;

    /* Flags to define presence of optional/expanded features */
    long tp_flags;

    char *tp_doc; /* Documentation string */

    /* Assigned meaning in release 2.0 */
    /* call function for all accessible objects */
    traverseproc tp_traverse;

    /* delete references to contained objects */
    inquiry tp_clear;

    /* Assigned meaning in release 2.1 */
    /* rich comparisons */
    richcmpfunc tp_richcompare;

    /* weak reference enabler */
    long tp_weaklistoffset;

    /* Added in release 2.2 */
    /* Iterators */
    getiterfunc tp_iter;
    iternextfunc tp_iternext;

    /* Attribute descriptor and subclassing stuff */
    struct PyMethodDef *tp_methods;
    struct PyMemberDef *tp_members;
    struct PyGetSetDef *tp_getset;
    struct _typeobject *tp_base;
    PyObject *tp_dict;
    descrgetfunc tp_descr_get;
    descrsetfunc tp_descr_set;
    long tp_dictoffset;
    initproc tp_init;
    allocfunc tp_alloc;
    newfunc tp_new;
    freefunc tp_free; /* Low-level free-memory routine */
    inquiry tp_is_gc; /* For PyObject_IS_GC */
    PyObject *tp_bases;
    PyObject *tp_mro; /* method resolution order */
    PyObject *tp_cache;
    PyObject *tp_subclasses;
    PyObject *tp_weaklist;

    } PyTypeObject;
    ```
+ _finalization and de-allocation_ 
    + `destructor tp_dealloc`
        + called when reference count is reduced to zero
        + used to free up memory 
        ```cpp 
        static void
        newdatatype_dealloc(newdatatypeobject * obj)
        {
            free(obj->obj_UnderlyingDatatypePtr);
            Py_TYPE(obj)->tp_free(obj);
        }
        ```
    + _deallocation leaves pending exception alone_
        + called frequently during stack unwind, usually due to exception
        + any action deallocator performs may cause additional python code to be executed, and may detect exception has been set 
        + may lead to missleading errrs
    + _solution_ 
        + save pending exception before performing unsafe action, restore it when done 
    ```cpp 
    static void
    my_dealloc(PyObject *obj)
    {
        MyObject *self = (MyObject *) obj;
        PyObject *cbresult;

        if (self->my_callback != NULL) {
            PyObject *err_type, *err_value, *err_traceback;
            int have_error = PyErr_Occurred() ? 1 : 0;

            if (have_error)
                PyErr_Fetch(&err_type, &err_value, &err_traceback);

            cbresult = PyObject_CallObject(self->my_callback, NULL);
            if (cbresult == NULL)
                PyErr_WriteUnraisable(self->my_callback);
            else
                Py_DECREF(cbresult);

            if (have_error)
                PyErr_Restore(err_type, err_value, err_traceback);

            Py_DECREF(self->my_callback);
        }
        Py_TYPE(obj)->tp_free((PyObject*)self);
    }
    ```
+ _object presentation_ 
    + 3 ways 
        + `reprfunc tp_repr` -> `repr()`
            + representation of instance
            ```cpp 
            static PyObject *
            newdatatype_repr(newdatatypeobject * obj)
            {
                return PyString_FromFormat("Repr-ified_newdatatype{{size:\%d}}",
                                        obj->obj_UnderlyingDatatypePtr->size);
            }
            ```
        + `reprfunc tp_str` -> `str()`
            + similar to `repr()` 
            + more human readable
            ```py 
            print node
            ```
            + _note_ 
                + `Py_PRINT_RAW` flag suggests print without stirng quotes 
            ```cpp 
            static int
            newdatatype_print(newdatatypeobject *obj, FILE *fp, int flags)
            {
                if (flags & Py_PRINT_RAW) {
                    fprintf(fp, "<{newdatatype object--size: %d}>",
                            obj->obj_UnderlyingDatatypePtr->size);
                }
                else {
                    fprintf(fp, "\"<{newdatatype object--size: %d}>\"",
                            obj->obj_UnderlyingDatatypePtr->size);
                }
                return 0;
            }
            ```
            + _note_ 
                + might want to save to file if `print` receives a file object...
        + `printfunc tp_print` -> `print()`
+ _attribute management_ 
    + 
    


    





```cpp 


// T = PyCoord, PyInterval, or similar
template <typename T>
DGPY_TYPE_BEGIN(Table)
	typedef typename T::table_type table_type;
	table_type* table;    // Instance of gene_table, exon_table, etc.
	INLINE static table_type& value(PyObject* self) { return *((PyTable<T>*)self)->table; }
	static PyMethodDef Methods[];
DGPY_TYPE_END



#define _DGPY_TYPEOBJ_BEGIN(tdecl, name, base, basicsize) \
	tdecl PyTypeObject Py##name::Type = {                       \
		PyObject_HEAD_INIT(NULL)                                \
		0,                         /*ob_size*/                  \
	};                                                          \
	tdecl PyTypeObject* Py##name::DefaultType = &Py##name::Type; \
	tdecl void Py##name::Register(PyObject* module)\
	{\
		EnsureInit(); \
		PyModule_AddObject(module, #name, (PyObject*)&Type);\
	}\
	tdecl void Py##name::EnsureInit()\
	{\
		if (!Type.ob_type) /* May have been previously initialized by a subtype's initializer */ \
			Init(); \
	}\
	tdecl void Py##name::Init()\
	{\
		typedef Py##name __py_type; DG_MARK_TYPE_USED(__py_type); \
		bool allow_smaller_basicsize = false; DG_MARK_USED(allow_smaller_basicsize); \
		PyTypeObject& _typeobj = Type;\
		const char* tp_name = DGPY_LIBNAME_STR "._cxx." #name;\
		Py_ssize_t tp_basicsize = basicsize;\
		Py_ssize_t tp_itemsize = 0;\
		destructor tp_dealloc = 0;\
		printfunc tp_print = 0;\
		getattrfunc tp_getattr = 0;\
		setattrfunc tp_setattr = 0;\
		cmpfunc tp_compare = 0;\
		reprfunc tp_repr = 0;\
		hashfunc tp_hash = 0;\
		ternaryfunc tp_call = 0;\
		reprfunc tp_str = 0;\
		getattrofunc tp_getattro = 0;\
		setattrofunc tp_setattro = 0;\
		PyBufferProcs* tp_as_buffer = 0;\
		long tp_flags = Py_TPFLAGS_DEFAULT;\
		const char* tp_doc = 0;\
		traverseproc tp_traverse = 0;\
		inquiry tp_clear = 0;\
		richcmpfunc tp_richcompare = 0;\
		Py_ssize_t tp_weaklistoffset = 0;\
		getiterfunc tp_iter = 0;\
		iternextfunc tp_iternext = 0;\
		PyMethodDef* tp_methods = 0;\
		PyMemberDef* tp_members = 0;\
		PyGetSetDef* tp_getset = 0;\
		PyTypeObject* tp_base = as_typeobject_ptr(base);\
		PyObject* tp_dict = 0;\
		descrgetfunc tp_descr_get = 0;\
		descrsetfunc tp_descr_set = 0;\
		Py_ssize_t tp_dictoffset = 0;\
		initproc tp_init = 0;\
		allocfunc tp_alloc = 0;\
		newfunc tp_new = 0;\
		freefunc tp_free = 0;\
		inquiry tp_is_gc = 0;\
		PyObject* tp_bases = 0;\
		PyObject* tp_mro = 0;\
		PyObject* tp_cache = 0;\
		PyObject* tp_subclasses = 0;\
		PyObject* tp_weaklist = 0;\
		destructor tp_del = 0;\
		\
		lenfunc sq_length = 0;\
		binaryfunc sq_concat = 0;\
		ssizeargfunc sq_repeat = 0;\
		ssizeargfunc sq_item = 0;\
		ssizessizeargfunc sq_slice = 0;\
		ssizeobjargproc sq_ass_item = 0;\
		ssizessizeobjargproc sq_ass_slice = 0;\
		objobjproc sq_contains = 0;\
		binaryfunc sq_inplace_concat = 0;\
		ssizeargfunc sq_inplace_repeat = 0;\
		\
		lenfunc mp_length = 0;\
		binaryfunc mp_subscript = 0;\
		objobjargproc mp_ass_subscript = 0;\
		{

#define DGPY_TYPEOBJ_END \
		}\
		if (tp_base && !allow_smaller_basicsize) \
			DG_CHECK(tp_base->tp_basicsize <= tp_basicsize, runtime, \
			         "subtype basicsize (%d) was less than basetype basicsize (%d); something went wrong", \
			         (int)tp_basicsize, (int)tp_base->tp_basicsize); \
		_typeobj.tp_name = tp_name;\
		_typeobj.tp_basicsize = tp_basicsize;\
		_typeobj.tp_itemsize = tp_itemsize;\
		_typeobj.tp_dealloc = tp_dealloc;\
		_typeobj.tp_print = tp_print;\
		_typeobj.tp_getattr = tp_getattr;\
		_typeobj.tp_setattr = tp_setattr;\
		_typeobj.tp_compare = tp_compare;\
		_typeobj.tp_repr = tp_repr;\
		_typeobj.tp_hash = tp_hash;\
		_typeobj.tp_call = tp_call;\
		_typeobj.tp_str = tp_str;\
		_typeobj.tp_getattro = tp_getattro;\
		_typeobj.tp_setattro = tp_setattro;\
		_typeobj.tp_as_buffer = tp_as_buffer;\
		_typeobj.tp_flags = tp_flags;\
		_typeobj.tp_doc = tp_doc;\
		_typeobj.tp_traverse = tp_traverse;\
		_typeobj.tp_clear = tp_clear;\
		_typeobj.tp_richcompare = tp_richcompare;\
		_typeobj.tp_weaklistoffset = tp_weaklistoffset;\
		_typeobj.tp_iter = tp_iter;\
		_typeobj.tp_iternext = tp_iternext;\
		_typeobj.tp_methods = tp_methods;\
		_typeobj.tp_members = tp_members;\
		_typeobj.tp_getset = tp_getset;\
		_typeobj.tp_base = tp_base;\
		_typeobj.tp_dict = tp_dict;\
		_typeobj.tp_descr_get = tp_descr_get;\
		_typeobj.tp_descr_set = tp_descr_set;\
		_typeobj.tp_dictoffset = tp_dictoffset;\
		_typeobj.tp_init = tp_init;\
		_typeobj.tp_alloc = tp_alloc;\
		_typeobj.tp_new = tp_new;\
		_typeobj.tp_free = tp_free;\
		_typeobj.tp_is_gc = tp_is_gc;\
		_typeobj.tp_bases = tp_bases;\
		_typeobj.tp_mro = tp_mro;\
		_typeobj.tp_cache = tp_cache;\
		_typeobj.tp_subclasses = tp_subclasses;\
		_typeobj.tp_weaklist = tp_weaklist;\
		_typeobj.tp_del = tp_del;\
		\
		static PySequenceMethods tp_as_sequence = {\
			sq_length,\
			sq_concat,\
			sq_repeat,\
			sq_item,\
			sq_slice,\
			sq_ass_item,\
			sq_ass_slice,\
			sq_contains,\
			sq_inplace_concat,\
			sq_inplace_repeat\
		};\
		_typeobj.tp_as_sequence = (sq_length || sq_concat || sq_repeat || sq_item || sq_slice || sq_ass_item || sq_ass_slice || sq_contains || sq_inplace_concat || sq_inplace_repeat) ? &tp_as_sequence : 0;\
		\
		static PyMappingMethods tp_as_mapping = {\
			mp_length,\
			mp_subscript,\
			mp_ass_subscript\
		};\
		_typeobj.tp_as_mapping = (mp_length || mp_subscript || mp_ass_subscript) ? &tp_as_mapping : 0;\
		PyType_Ready(&_typeobj);\
		if (tp_base) \
			PyForceNewGCInheritence(&_typeobj); /* Adopt newer (smarter) inheritence rules for GC tracking */ \
	}
```