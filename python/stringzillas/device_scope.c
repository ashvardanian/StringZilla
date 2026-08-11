/**
 *  @brief Execution context selecting the CPU or GPU hardware the engines run on.
 *  @file python/stringzillas/device_scope.c
 *  @author Ash Vardanian
 */
#include "stringzillas.h"

static void DeviceScope_dealloc(DeviceScope *self) {
    if (self->handle) {
        szs_device_scope_free(self->handle);
        self->handle = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *DeviceScope_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    DeviceScope *self = (DeviceScope *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->handle = NULL;
        self->description[0] = '\0';
    }
    return (PyObject *)self;
}

static int DeviceScope_init(DeviceScope *self, PyObject *args, PyObject *kwargs) {
    sz_size_t cpu_cores = 0;
    sz_size_t gpu_device = 0;
    PyObject *cpu_cores_obj = NULL;
    PyObject *gpu_device_obj = NULL;

    static char *kwlist[] = {"cpu_cores", "gpu_device", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", kwlist, &cpu_cores_obj, &gpu_device_obj)) return -1;

    sz_status_t status;
    char const *error_detail = NULL;

    if (cpu_cores_obj != NULL && gpu_device_obj != NULL) {
        PyErr_SetString(PyExc_ValueError, "Cannot specify both cpu_cores and gpu_device");
        return -1;
    }
    else if (cpu_cores_obj != NULL) {
        if (!PyLong_Check(cpu_cores_obj)) {
            PyErr_SetString(PyExc_TypeError, "cpu_cores must be an integer");
            return -1;
        }
        cpu_cores = PyLong_AsSize_t(cpu_cores_obj);
        if (cpu_cores == (sz_size_t)-1 && PyErr_Occurred()) { return -1; }
        status = szs_device_scope_init_cpu_cores(cpu_cores, &self->handle, &error_detail);
        if (cpu_cores == 1) { snprintf(self->description, sizeof(self->description), "default"); }
        else if (cpu_cores == 0) { snprintf(self->description, sizeof(self->description), "CPUs:all"); }
        else { snprintf(self->description, sizeof(self->description), "CPUs:%zu", cpu_cores); }
    }
    else if (gpu_device_obj != NULL) {
        if (!PyLong_Check(gpu_device_obj)) {
            PyErr_SetString(PyExc_TypeError, "gpu_device must be an integer");
            return -1;
        }
        gpu_device = PyLong_AsSize_t(gpu_device_obj);
        if (gpu_device == (sz_size_t)-1 && PyErr_Occurred()) { return -1; }
        status = szs_device_scope_init_gpu_device(gpu_device, &self->handle, &error_detail);
        snprintf(self->description, sizeof(self->description), "GPU:%zu", gpu_device);
    }
    else {
        status = szs_device_scope_init_default(&self->handle, &error_detail);
        snprintf(self->description, sizeof(self->description), "default");
    }

    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "DeviceScope initialization");
        return -1;
    }

    return 0;
}

static PyObject *DeviceScope_repr(DeviceScope *self) {
    return PyUnicode_FromFormat("DeviceScope(%s)", self->description);
}

static char const doc_DeviceScope[] =                                                   //
    "DeviceScope(cpu_cores=None, gpu_device=None)\n"                                    //
    "\n"                                                                                //
    "Context for controlling execution on CPU cores or GPU devices.\n"                  //
    "\n"                                                                                //
    "Args:\n"                                                                           //
    "  cpu_cores (int, optional): Number of CPU cores to use, or zero for all cores.\n" //
    "  gpu_device (int, optional): GPU device ID to target.\n"                          //
    "\n"                                                                                //
    "Note: Cannot specify both cpu_cores and gpu_device.\n"                             //
    "\n"                                                                                //
    "Examples:\n"                                                                       //
    "  >>> import stringzillas as szs\n"                                                //
    "  >>> scope = szs.DeviceScope(cpu_cores=4)  # restrict engines to 4 CPU cores";

PyTypeObject DeviceScopeType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.DeviceScope",
    .tp_doc = doc_DeviceScope,
    .tp_basicsize = sizeof(DeviceScope),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = DeviceScope_new,
    .tp_init = (initproc)DeviceScope_init,
    .tp_dealloc = (destructor)DeviceScope_dealloc,
    .tp_repr = (reprfunc)DeviceScope_repr,
};
