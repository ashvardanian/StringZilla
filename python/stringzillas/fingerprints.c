/**
 *  @brief Min-Hash and Count-Min-Sketch fingerprinting engines.
 *  @file python/stringzillas/fingerprints.c
 *  @author Ash Vardanian
 */
#include "stringzillas.h"

/**
 *  @brief  Fingerprinting engine for binary strings.
 */
typedef struct {
    PyObject ob_base;
    szs_fingerprints_t handle;
    char description[64];
    sz_capability_t capabilities;
    sz_size_t ndim;
    SZS_LOCK_FIELD_
} Fingerprints;

static void Fingerprints_dealloc(Fingerprints *self) {
    if (self->handle) {
        szs_fingerprints_free(self->handle);
        self->handle = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Fingerprints_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    Fingerprints *self = (Fingerprints *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->handle = NULL;
        self->description[0] = '\0';
        self->capabilities = 0;
        self->ndim = 0;
    }
    return (PyObject *)self;
}

static int Fingerprints_init(Fingerprints *self, PyObject *args, PyObject *kwargs) {
    sz_size_t ndim;
    PyObject *window_widths_obj = NULL;
    sz_size_t alphabet_size = 256;
    unsigned long long seed = 0;
    PyObject *capabilities_tuple = NULL;
    sz_capability_t capabilities = active_capabilities_;

    static char *kwlist[] = {"ndim", "window_widths", "alphabet_size", "seed", "capabilities", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|OnKO", kwlist, &ndim, &window_widths_obj, &alphabet_size, &seed,
                                     &capabilities_tuple))
        return -1;

    // Parse capabilities if provided
    if (capabilities_tuple)
        if (parse_and_intersect_capabilities(capabilities_tuple, &capabilities) != 0) return -1;

    sz_size_t *window_widths = NULL;
    sz_size_t window_widths_count = 0;

    // Parse window_widths if provided - require NumPy array of uint64
    if (window_widths_obj && window_widths_obj != Py_None) {
        if (!PyArray_Check(window_widths_obj)) {
            PyErr_SetString(PyExc_TypeError, "window_widths must be a numpy array of uint64");
            return -1;
        }

        PyArrayObject *arr = (PyArrayObject *)window_widths_obj;

        // Check dtype is uint64
        if (PyArray_TYPE(arr) != NPY_UINT64) {
            PyErr_SetString(PyExc_TypeError, "window_widths must have dtype uint64");
            return -1;
        }

        // Check that it's 1D
        if (PyArray_NDIM(arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "window_widths must be a 1D array");
            return -1;
        }

        // Check that it's contiguous (no strides)
        if (!PyArray_IS_C_CONTIGUOUS(arr)) {
            PyErr_SetString(PyExc_ValueError, "window_widths must be a contiguous C-style array (no strides)");
            return -1;
        }

        window_widths_count = PyArray_SIZE(arr);
        window_widths = (sz_size_t *)PyArray_DATA(arr);
    }

    char const *error_detail = NULL;
    sz_status_t status = szs_fingerprints_init(ndim, alphabet_size, window_widths, window_widths_count, (sz_u64_t)seed,
                                               NULL, capabilities, &self->handle, &error_detail);

    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "Fingerprints initialization");
        return -1;
    }

    snprintf(self->description, sizeof(self->description), "ndim=%zu,window_widths=%zu,alphabet_size=%zu,seed=%llu",
             ndim, window_widths_count, alphabet_size, seed);
    self->capabilities = capabilities;
    self->ndim = ndim;
    return 0;
}

static PyObject *Fingerprints_repr(Fingerprints *self) {
    return PyUnicode_FromFormat("Fingerprints(%s)", self->description);
}

static PyObject *Fingerprints_get_capabilities(Fingerprints *self, void *closure) {
    return capabilities_to_tuple(self->capabilities);
}

static PyObject *Fingerprints_call(Fingerprints *self, PyObject *args, PyObject *kwargs) {

    PyObject *texts_obj = NULL, *device_obj = NULL, *out_obj = NULL;
    static char *kwlist[] = {"texts", "device", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO", kwlist, &texts_obj, &device_obj, &out_obj)) return NULL;

    DeviceScope *device_scope = NULL;
    if (device_obj != NULL && device_obj != Py_None) {
        if (!PyObject_TypeCheck(device_obj, &DeviceScopeType)) {
            PyErr_SetString(PyExc_TypeError, "device must be a DeviceScope instance");
            return NULL;
        }
        device_scope = (DeviceScope *)device_obj;
    }

    szs_device_scope_t device_handle = device_scope ? device_scope->handle : default_device_scope;

    // Handle empty input - return tuple of empty arrays
    if (PySequence_Check(texts_obj) && PySequence_Size(texts_obj) == 0) {
        npy_intp dims[2] = {0, self->ndim};
        PyArrayObject *empty_hashes = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_UINT32);
        PyArrayObject *empty_counts = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_UINT32);

        if (!empty_hashes || !empty_counts) {
            Py_XDECREF(empty_hashes);
            Py_XDECREF(empty_counts);
            return PyErr_NoMemory();
        }

        PyObject *result_tuple = PyTuple_New(2);
        if (!result_tuple) {
            Py_DECREF(empty_hashes);
            Py_DECREF(empty_counts);
            return NULL;
        }

        PyTuple_SET_ITEM(result_tuple, 0, (PyObject *)empty_hashes);
        PyTuple_SET_ITEM(result_tuple, 1, (PyObject *)empty_counts);
        return result_tuple;
    }

    // Swap allocators only when using CUDA with a GPU device (inputs must be unified)
    sz_bool_t need_unified = requires_device_memory(self->capabilities);
    if (need_unified)
        if (!try_swap_to_unified_allocator(texts_obj)) return NULL;

    sz_size_t kernel_input_size = 0;
    void *kernel_texts_punned = NULL;
    sz_status_t (*kernel_punned)(szs_fingerprints_t, szs_device_scope_t, void *, sz_u32_t *, sz_size_t, sz_u32_t *,
                                 sz_size_t, char const **) = NULL;

    // Handle 32-bit tape inputs
    sz_sequence_u32tape_t texts_u32tape;
    sz_bool_t texts_is_u32tape = sz_py_export_strings_as_u32tape( //
        texts_obj, &texts_u32tape.data, &texts_u32tape.offsets, &texts_u32tape.count);
    if (texts_is_u32tape) {
        kernel_input_size = texts_u32tape.count;
        kernel_punned = szs_fingerprints_u32tape;
        kernel_texts_punned = &texts_u32tape;
    }

    // Handle 64-bit tape inputs
    sz_sequence_u64tape_t texts_u64tape;
    sz_bool_t texts_is_u64tape = !texts_is_u32tape &&
                                 sz_py_export_strings_as_u64tape( //
                                     texts_obj, &texts_u64tape.data, &texts_u64tape.offsets, &texts_u64tape.count);
    if (texts_is_u64tape) {
        kernel_input_size = texts_u64tape.count;
        kernel_punned = szs_fingerprints_u64tape;
        kernel_texts_punned = &texts_u64tape;
    }

    // Handle generic sequence inputs
    sz_sequence_t texts_seq;
    sz_bool_t texts_is_sequence = !texts_is_u32tape && !texts_is_u64tape &&
                                  sz_py_export_strings_as_sequence(texts_obj, &texts_seq);
    if (texts_is_sequence) {
        kernel_input_size = texts_seq.count;
        kernel_punned = szs_fingerprints_sequence;
        kernel_texts_punned = &texts_seq;
    }

    if (kernel_punned == NULL) {
        PyErr_Format(PyExc_TypeError,
                     "Expected stringzilla.Strs object, got %s. Convert using: stringzilla.Strs(your_string_list)",
                     Py_TYPE(texts_obj)->tp_name);
        return NULL;
    }

    // Create NumPy outputs up front and copy into them (CPU or GPU)
    npy_intp dims[2] = {kernel_input_size, self->ndim};
    PyArrayObject *hashes_array = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_UINT32);
    PyArrayObject *counts_array = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_UINT32);
    if (!hashes_array || !counts_array) {
        Py_XDECREF(hashes_array);
        Py_XDECREF(counts_array);
        return PyErr_NoMemory();
    }

    // Determine bytes to write; if zero, we'll just return the empty arrays
    sz_memory_allocator_t *out_alloc = need_unified ? &unified_allocator : &default_allocator;
    sz_size_t const total_elements = kernel_input_size * self->ndim;
    sz_size_t const total_bytes = total_elements * sizeof(sz_u32_t);

    if (total_bytes > 0) {
        sz_u32_t *buf_hashes = (sz_u32_t *)out_alloc->allocate(total_bytes, out_alloc->handle);
        sz_u32_t *buf_counts = (sz_u32_t *)out_alloc->allocate(total_bytes, out_alloc->handle);
        if (!buf_hashes || !buf_counts) {
            if (buf_hashes) out_alloc->free(buf_hashes, total_bytes, out_alloc->handle);
            if (buf_counts) out_alloc->free(buf_counts, total_bytes, out_alloc->handle);
            Py_DECREF(hashes_array);
            Py_DECREF(counts_array);
            return PyErr_NoMemory();
        }

        char const *error_detail = NULL;
        if (device_scope) SZS_LOCK_(&device_scope->lock);
        SZS_LOCK_(&self->lock);
        sz_status_t status = kernel_punned(self->handle, device_handle, kernel_texts_punned, buf_hashes,
                                           self->ndim * sizeof(sz_u32_t), buf_counts, self->ndim * sizeof(sz_u32_t),
                                           &error_detail);
        SZS_UNLOCK_(&self->lock);
        if (device_scope) SZS_UNLOCK_(&device_scope->lock);
        if (status != sz_success_k) {
            out_alloc->free(buf_hashes, total_bytes, out_alloc->handle);
            out_alloc->free(buf_counts, total_bytes, out_alloc->handle);
            Py_DECREF(hashes_array);
            Py_DECREF(counts_array);
            set_stringzilla_error(status, error_detail, "Fingerprints computation");
            return NULL;
        }

        memcpy(PyArray_DATA(hashes_array), buf_hashes, total_bytes);
        memcpy(PyArray_DATA(counts_array), buf_counts, total_bytes);
        out_alloc->free(buf_hashes, total_bytes, out_alloc->handle);
        out_alloc->free(buf_counts, total_bytes, out_alloc->handle);
    }

    PyObject *result_tuple = PyTuple_New(2);
    if (!result_tuple) {
        Py_DECREF(hashes_array);
        Py_DECREF(counts_array);
        return NULL;
    }

    PyTuple_SET_ITEM(result_tuple, 0, (PyObject *)hashes_array);
    PyTuple_SET_ITEM(result_tuple, 1, (PyObject *)counts_array);

    return result_tuple;
}

static char const doc_Fingerprints[] =                                                                               //
    "Fingerprints(ndim, window_widths=None, alphabet_size=256, seed=0, capabilities=None)\n"                         //
    "\n"                                                                                                             //
    "Compute MinHash fingerprints for binary strings.\n"                                                             //
    "\n"                                                                                                             //
    "Args:\n"                                                                                                        //
    "  ndim (int): Number of dimensions per fingerprint.\n"                                                          //
    "  window_widths (numpy.array, optional): 1D uint64 contiguous array of window widths. Uses defaults if None.\n" //
    "  alphabet_size (int, optional): Alphabet size, default 256 for binary strings.\n"                              //
    "  seed (int, optional): Reproducibility seed; every value derives independent per-dimension multipliers and\n"  //
    "                        moduli for MinHash independence (0 is the default seed, not a special mode).\n"         //
    "  capabilities (Tuple[str] or DeviceScope, optional): Hardware capabilities to use.\n"                          //
    "                                       Can be explicit capabilities like ('serial', 'parallel', 'cuda')\n"      //
    "                                       or a DeviceScope for automatic capability inference.\n"                  //
    "\n"                                                                                                             //
    "Call with:\n"                                                                                                   //
    "  texts (sequence): Sequence of strings to fingerprint.\n"                                                      //
    "  device (DeviceScope, optional): Device execution context.\n"                                                  //
    "\n"                                                                                                             //
    "Returns:\n"                                                                                                     //
    "  tuple: (hashes_matrix, counts_matrix) - Two numpy uint32 matrices of shape (num_texts, ndim).\n"              //
    "\n"                                                                                                             //
    "Examples:\n"                                                                                                    //
    "  >>> # Minimal CPU example with auto-inferred capabilities\n"                                                  //
    "  >>> import stringzilla as sz, stringzillas as szs\n"                                                          //
    "  >>> engine = szs.Fingerprints(ndim=128)\n"                                                                    //
    "  >>> docs = sz.Strs(['document one', 'document two', 'document three'])\n"                                     //
    "  >>> hashes, counts = engine(docs)\n"                                                                          //
    "  >>> # GPU example with custom dimensions; falls back to CPU when CUDA is unavailable\n"                       //
    "  >>> scope = szs.DeviceScope(gpu_device=0) if 'cuda' in szs.__capabilities__ else szs.DeviceScope()\n"         //
    "  >>> engine = szs.Fingerprints(ndim=256, capabilities=scope)\n"                                                //
    "  >>> hashes, counts = engine(docs, device=scope)";
static PyGetSetDef Fingerprints_getsetters[] = {
    {"__capabilities__", (getter)Fingerprints_get_capabilities, NULL, doc_capabilities, NULL}, //
    {NULL}                                                                                     /* Sentinel */
};

PyTypeObject FingerprintsType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.Fingerprints",
    .tp_doc = doc_Fingerprints,
    .tp_basicsize = sizeof(Fingerprints),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Fingerprints_new,
    .tp_init = (initproc)Fingerprints_init,
    .tp_getset = Fingerprints_getsetters,
    .tp_repr = (reprfunc)Fingerprints_repr,
    .tp_dealloc = (destructor)Fingerprints_dealloc,
    .tp_call = (ternaryfunc)Fingerprints_call,
};
