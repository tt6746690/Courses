#include <Python.h>


// custom exception object
static PyObject *SpamError;

static PyObject *spam_system(PyObject *self, PyObject *args){
    const char*command;
    int sts;

    if(!PyArg_ParseTuple(args, "s, &command"))
        return NULL;

    sts = system(command);
    if(sts < 0){
        PyErr_SetString(SpamError, "System command failed");
        return NULL;
    }

    return Py_BuildValue("i", sts);     // returns an integer value
}



PyMODINIT_FUNC initspam(void)
{
    PyObject *m;

    m = Py_InitModule("spam", SpamMethods);
    if (m == NULL)
        return;

    SpamError = PyErr_NewException("spam.error", NULL, NULL);
    Py_INCREF(SpamError);
    PyModule_AddObject(m, "error", SpamError);
}


