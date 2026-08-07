/**
 *  @brief The `Str` type - construction, buffer protocol, and byte-level transforms.
 *  @file python/stringzilla/str.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

static int Str_init(Str *self, PyObject *args, PyObject *kwargs) {

    // Parse all arguments into PyObjects first
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs > 3) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return -1;
    }
    PyObject *parent_obj = nargs >= 1 ? PyTuple_GET_ITEM(args, 0) : NULL;
    PyObject *from_obj = nargs >= 2 ? PyTuple_GET_ITEM(args, 1) : NULL;
    PyObject *to_obj = nargs >= 3 ? PyTuple_GET_ITEM(args, 2) : NULL;

    // Parse keyword arguments, if provided, and ensure no duplicates
    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value))
            if (PyUnicode_CompareWithASCIIString(key, "parent") == 0 && !parent_obj) { parent_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "from") == 0 && !from_obj) { from_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "to") == 0 && !to_obj) { to_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) return -1;
    }

    // Now, layout-check and cast each argument
    Py_ssize_t from = 0, to = PY_SSIZE_T_MAX;
    if (from_obj) {
        from = PyLong_AsSsize_t(from_obj);
        if (from == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The `from` argument must be an integer");
            return -1;
        }
    }
    if (to_obj) {
        to = PyLong_AsSsize_t(to_obj);
        if (to == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The `to` argument must be an integer");
            return -1;
        }
    }

    // Re-initialization: release any state bound by a prior __init__ before rebinding (mirrors Str_dealloc)
    if (self->parent) { Py_CLEAR(self->parent); }
    else if (self->memory.start) { free((void *)self->memory.start); }
    self->memory.start = NULL;
    self->memory.length = 0;

    // Handle empty string
    if (parent_obj == NULL) {
        self->memory.start = NULL;
        self->memory.length = 0;
    }
    // Increment the reference count of the parent
    else if (sz_py_export_string_like(parent_obj, &self->memory.start, &self->memory.length)) {
        self->parent = parent_obj;
        Py_INCREF(parent_obj);
    }
    else {
        wrap_current_exception("Unsupported parent type");
        return -1;
    }

    // Apply slicing
    sz_size_t normalized_offset, normalized_length;
    sz_ssize_clamp_interval(self->memory.length, from, to, &normalized_offset, &normalized_length);
    self->memory.start = ((sz_ptr_t)self->memory.start) + normalized_offset;
    self->memory.length = normalized_length;
    return 0;
}

PyObject *Str_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Str *self = Str_alloc_(); // `Str` is not a base type, so `type` is always `&StrType`.
    if (!self) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't allocate a Str handle!");
        return NULL;
    }
    return (PyObject *)self;
}

static void Str_dealloc(Str *self) {
    if (self->parent) { Py_XDECREF(self->parent); }
    else if (self->memory.start) { free(self->memory.start); }

    // Park the dead header on the free-list instead of freeing it, threading the `next` link through
    // the now-unused `parent` field; fall through to a real free when the cache is full or absent.
    stringzilla_state_t *state = stringzilla_state_();
    if (state) sz_freelist_lock_(state);
    if (state && state->str_freelist_count < sz_freelist_capacity_k) {
        self->parent = (PyObject *)state->str_freelist_head;
        state->str_freelist_head = self;
        state->str_freelist_count++;
        sz_freelist_unlock_(state);
        return;
    }
    if (state) sz_freelist_unlock_(state);
    self->parent = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Str_str(Str *self) { return PyUnicode_FromStringAndSize(self->memory.start, self->memory.length); }

static PyObject *Str_repr(Str *self) {
    // Interestingly, known-length string formatting only works in Python 3.12 and later.
    // https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_FromFormat
    if (PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 12)
        return PyUnicode_FromFormat("sz.Str('%.*s')", (int)self->memory.length, self->memory.start);
    else {
        // Use a simpler formatting rule for older versions
        PyObject *str_obj = PyUnicode_FromStringAndSize(self->memory.start, self->memory.length);
        PyObject *result = PyUnicode_FromFormat("sz.Str('%U')", str_obj);
        Py_DECREF(str_obj);
        return result;
    }
}

static PyObject *Str_get_address(Str *self, void *closure) { return PyLong_FromSize_t((sz_size_t)self->memory.start); }
static PyObject *Str_get_nbytes(Str *self, void *closure) { return PyLong_FromSize_t(self->memory.length); }

static Py_ssize_t Str_len(Str *self) { return self->memory.length; }

static PyObject *Str_getitem(Str *self, Py_ssize_t i) {

    // Negative indexing
    if (i < 0) i += self->memory.length;

    if (i < 0 || (sz_size_t)i >= self->memory.length) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }

    // Assuming the underlying data is UTF-8 encoded
    return PyUnicode_FromStringAndSize(self->memory.start + i, 1);
}

static PyObject *Str_subscript(Str *self, PyObject *key) {
    if (PySlice_Check(key)) {
        // Sanity checks
        Py_ssize_t start, stop, step;
        if (PySlice_Unpack(key, &start, &stop, &step) < 0) return NULL;
        // Keep the reported length: for an inverted window (e.g. `s[3:1]`) `stop - start` is negative.
        Py_ssize_t const slice_length = PySlice_AdjustIndices(self->memory.length, &start, &stop, step);
        if (step != 1) {
            PyErr_SetString(PyExc_IndexError, "Efficient step is not supported");
            return NULL;
        }

        // Create a new `Str` object
        Str *self_slice = Str_alloc_();
        if (self_slice == NULL) return PyErr_NoMemory();

        // Set its properties based on the slice
        self_slice->memory.start = self->memory.start + start;
        self_slice->memory.length = slice_length;
        self_slice->parent = (PyObject *)self; // Set parent to keep it alive

        // Increment the reference count of the parent
        Py_INCREF(self);
        return (PyObject *)self_slice;
    }
    else if (PyLong_Check(key)) {
        Py_ssize_t index = PyLong_AsSsize_t(key);
        if (index == -1 && PyErr_Occurred()) return NULL; // Propagate OverflowError instead of leaking it
        return Str_getitem(self, index);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Str indices must be integers or slices");
        return NULL;
    }
}

static int Str_getbuffer(Str *self, Py_buffer *view, int flags) {
    if (view == NULL) {
        PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
        return -1;
    }

    // The backing bytes are immutable (str/bytes/mmap), so refuse a writable request instead of
    // silently handing back a "writable" view over read-only memory.
    if ((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE) {
        PyErr_SetString(PyExc_BufferError, "Str is read-only");
        return -1;
    }

    static Py_ssize_t itemsize[1] = {1};
    view->obj = (PyObject *)self;
    view->buf = self->memory.start;
    view->len = self->memory.length;
    view->readonly = 1;
    view->itemsize = sizeof(char);
    // Only advertise a format string when the consumer asked for one
    view->format = (flags & PyBUF_FORMAT) ? "c"
                                          : NULL; // https://docs.python.org/3/library/struct.html#format-characters
    view->ndim = 1;
    view->shape = (Py_ssize_t *)&self->memory.length; // 1-D array, so shape is just a pointer to the length
    view->strides = itemsize;                         // strides in a 1-D array is just the item size
    view->suboffsets = NULL;
    view->internal = NULL;

    Py_INCREF(self);
    return 0;
}

static void Str_releasebuffer(PyObject *_, Py_buffer *view) {
    //! This function MUST NOT decrement view->obj, since that is done automatically
    //! in PyBuffer_Release() (this scheme is useful for breaking reference cycles).
    //! https://docs.python.org/3/c-api/typeobj.html#c.PyBufferProcs.bf_releasebuffer
}

char const doc_decode[] =                                                      //
    "Decode the bytes into a Unicode string with a given encoding.\n"          //
    "\n"                                                                       //
    "Args:\n"                                                                  //
    "  text (Str or str or bytes): The string object.\n"                       //
    "  encoding (str, optional): The encoding to use (default is 'utf-8').\n"  //
    "  errors (str, optional): Error handling scheme (default is 'strict').\n" //
    "Returns:\n"                                                               //
    "  str: The decoded Unicode string.\n"                                     //
    "Raises:\n"                                                                //
    "  UnicodeDecodeError: If decoding fails.\n"                               //
    "\n"                                                                       //
    "Example:\n"                                                               //
    "  >>> sz.Str('abc').decode() == 'abc'\n"                                  //
    "  True";

PyObject *Str_like_decode(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                          PyObject *args_names_tuple) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count < !is_member || positional_args_count > !is_member + 2) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    PyObject *encoding_obj = positional_args_count > !is_member + 0 ? args[!is_member + 0] : NULL;
    PyObject *errors_obj = positional_args_count > !is_member + 1 ? args[!is_member + 1] : NULL;

    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "encoding") == 0 && !encoding_obj) { encoding_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "errors") == 0 && !errors_obj) { errors_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) return NULL;
        }
    }

    // Convert `encoding` and `errors` to `NULL` if they are `None`
    if (encoding_obj == Py_None) encoding_obj = NULL;
    if (errors_obj == Py_None) errors_obj = NULL;

    sz_string_view_t text, encoding, errors;
    if ((!sz_py_export_string_like(text_obj, &text.start, &text.length)) ||
        (encoding_obj && !sz_py_export_string_like(encoding_obj, &encoding.start, &encoding.length)) ||
        (errors_obj && !sz_py_export_string_like(errors_obj, &errors.start, &errors.length))) {
        wrap_current_exception("text, encoding, and errors must be string-like");
        return NULL;
    }

    if (encoding_obj == NULL) encoding = (sz_string_view_t) {"utf-8", 5};
    if (errors_obj == NULL) errors = (sz_string_view_t) {"strict", 6};

    // Python docs: https://docs.python.org/3/library/stdtypes.html#bytes.decode
    // CPython docs: https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_Decode
    return PyUnicode_Decode(text.start, text.length, encoding.start, errors.start);
}

char const doc_write_to[] =                                             //
    "Write the string to a file.\n"                                     //
    "\n"                                                                //
    "Args:\n"                                                           //
    "  text (Str or str or bytes): The string object.\n"                //
    "  filename (str): The file path to write to.\n"                    //
    "Returns:\n"                                                        //
    "  None.\n"                                                         //
    "\n"                                                                //
    "Example:\n"                                                        //
    "  >>> import os, tempfile\n"                                       //
    "  >>> path = os.path.join(tempfile.gettempdir(), 'sz_demo.bin')\n" //
    "  >>> sz.Str('payload').write_to(path)";

PyObject *Str_write_to(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                       PyObject *args_names_tuple) {

    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count != !is_member + 1) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    PyObject *path_obj = args[!is_member + 0];

    // Parse keyword arguments
    if (args_names_tuple) {
        PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument");
        return NULL;
    }

    sz_string_view_t text;
    sz_string_view_t path;

    // Validate and convert `text` and `path`
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length) ||
        !sz_py_export_string_like(path_obj, &path.start, &path.length)) {
        wrap_current_exception("Text and path must be string-like");
        return NULL;
    }

    // An embedded NUL would silently truncate the path at `fopen`; reject it like CPython's open()
    if (sz_find_byte(path.start, path.length, "\0") != NULL) {
        PyErr_SetString(PyExc_ValueError, "embedded null byte in path");
        return NULL;
    }

    // There is a chance, the path isn't NULL-terminated, so copy it to a new buffer.
    // Many OSes have fairly low limit for the maximum path length.
    // On Windows its 260, but up to __around__ 32,767 characters are supported in extended API.
    // But it's better to be safe than sorry and use malloc :)
    //
    // https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry
    // https://doc.owncloud.com/server/next/admin_manual/troubleshooting/path_filename_length.html
    sz_ptr_t path_buffer = (sz_ptr_t)malloc(path.length + 1);
    if (path_buffer == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for the path");
        return NULL;
    }
    sz_copy(path_buffer, path.start, path.length);
    path_buffer[path.length] = '\0';

    // Unlock the Global Interpreter Lock (GIL) to allow other threads to run
    // while the current thread is waiting for the file to be written.
    PyThreadState *gil_state = PyEval_SaveThread();
    FILE *file_pointer = fopen(path_buffer, "wb");
    if (file_pointer == NULL) {
        PyEval_RestoreThread(gil_state);
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path_buffer);
        free(path_buffer);
        return NULL;
    }

    setbuf(file_pointer, NULL); // Set the stream to unbuffered
    int status = fwrite(text.start, 1, text.length, file_pointer);
    PyEval_RestoreThread(gil_state);
    if (status != (Py_ssize_t)text.length) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path_buffer);
        free(path_buffer);
        fclose(file_pointer);
        return NULL;
    }

    free(path_buffer);
    fclose(file_pointer);
    Py_RETURN_NONE;
}

char const doc_offset_within[] =                                                                  //
    "Return the raw byte offset of this StringZilla string within a larger StringZilla string.\n" //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  text (Str or str or bytes): The substring.\n"                                              //
    "  larger (Str): The larger string to search within.\n"                                       //
    "Returns:\n"                                                                                  //
    "  int: The byte offset where 'self' is found within 'larger', or -1 if not found.\n"         //
    "\n"                                                                                          //
    "Example:\n"                                                                                  //
    "  >>> big = sz.Str('hello world')\n"                                                         //
    "  >>> big[6:].offset_within(big)\n"                                                          //
    "  6";

PyObject *Str_offset_within(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                            PyObject *args_names_tuple) {

    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count != !is_member + 1) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *slice_obj = is_member ? self : args[0];
    PyObject *text_obj = args[!is_member + 0];

    // Parse keyword arguments
    if (args_names_tuple) {
        PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument");
        return NULL;
    }

    sz_string_view_t text;
    sz_string_view_t slice;

    // Validate and convert `text` and `slice`
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length) ||
        !sz_py_export_string_like(slice_obj, &slice.start, &slice.length)) {
        wrap_current_exception("Text and slice must be string-like");
        return NULL;
    }

    // Pointer-based containment check; intended for actual parent/child slices. Returns -1 (per the
    // docstring) when the slice does not sit within the text, rather than raising.
    if (slice.start < text.start || slice.start + slice.length > text.start + text.length)
        return PyLong_FromSsize_t(-1);

    return PyLong_FromSize_t((sz_size_t)(slice.start - text.start));
}

static PyObject *Str_concat(PyObject *self, PyObject *other) {
    struct sz_string_view_t self_str, other_str;

    // Validate and convert `self` and `other`
    if (!sz_py_export_string_like(self, &self_str.start, &self_str.length) ||
        !sz_py_export_string_like(other, &other_str.start, &other_str.length)) {
        wrap_current_exception("Both operands must be string-like");
        return NULL;
    }

    // Allocate a new Str instance
    Str *result_str = Str_alloc_();
    if (result_str == NULL) { return NULL; }

    // Calculate the total length of the new string
    result_str->parent = NULL;
    result_str->memory.length = self_str.length + other_str.length;

    // Allocate memory for the new string
    result_str->memory.start = malloc(result_str->memory.length);
    if (result_str->memory.start == NULL) {
        Py_DECREF(result_str);
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for string concatenation");
        return NULL;
    }

    // Perform the string concatenation
    sz_copy(result_str->memory.start, self_str.start, self_str.length);
    sz_copy(result_str->memory.start + self_str.length, other_str.start, other_str.length);

    return (PyObject *)result_str;
}

static PySequenceMethods Str_as_sequence = {
    .sq_length = Str_len,   //
    .sq_item = Str_getitem, //
    .sq_contains = Str_in,  //
};

static PyMappingMethods Str_as_mapping = {
    .mp_length = Str_len,          //
    .mp_subscript = Str_subscript, // Is used to implement slices in Python
};

static PyBufferProcs Str_as_buffer = {
    .bf_getbuffer = Str_getbuffer,
    .bf_releasebuffer = Str_releasebuffer,
};

static PyNumberMethods Str_as_number = {
    .nb_add = Str_concat,
};

static char const doc_Str_address[] =                                  //
    "Memory address of the first byte of the string, as an integer.\n" //
    "\n"                                                               //
    "Enables zero-copy interop (e.g. with PyArrow or ctypes).\n"       //
    "\n"                                                               //
    "Example:\n"                                                       //
    "  >>> isinstance(sz.Str('abc').address, int)\n"                   //
    "  True";

static char const doc_Str_nbytes[] =   //
    "Length of the string in bytes.\n" //
    "\n"                               //
    "Example:\n"                       //
    "  >>> sz.Str('abc').nbytes\n"     //
    "  3";

static PyGetSetDef Str_getsetters[] = {
    // Compatibility with PyArrow
    {"address", (getter)Str_get_address, NULL, doc_Str_address, NULL}, //
    {"nbytes", (getter)Str_get_nbytes, NULL, doc_Str_nbytes, NULL},    //
    {NULL}                                                             // Sentinel
};

static PyMethodDef Str_methods[] = {
    {"contains", (PyCFunction)Str_like_contains, SZ_METHOD_FLAGS, doc_contains},
    {"count", (PyCFunction)Str_like_count, SZ_METHOD_FLAGS, doc_count},
    {"splitlines", (PyCFunction)Str_like_splitlines, SZ_METHOD_FLAGS, doc_splitlines},
    {"startswith", (PyCFunction)Str_like_startswith, SZ_METHOD_FLAGS, doc_startswith},
    {"endswith", (PyCFunction)Str_like_endswith, SZ_METHOD_FLAGS, doc_endswith},
    {"decode", (PyCFunction)Str_like_decode, SZ_METHOD_FLAGS, doc_decode},
    {"hash", (PyCFunction)Str_like_hash, SZ_METHOD_FLAGS, doc_like_hash},
    {"hash_multiseed", (PyCFunction)Str_like_hash_multiseed, SZ_METHOD_FLAGS, doc_hash_multiseed},
    {"sha256", (PyCFunction)Str_like_sha256, SZ_METHOD_FLAGS, doc_like_sha256},
    {"lstrip", (PyCFunction)Str_like_lstrip, SZ_METHOD_FLAGS, doc_lstrip},
    {"rstrip", (PyCFunction)Str_like_rstrip, SZ_METHOD_FLAGS, doc_rstrip},
    {"strip", (PyCFunction)Str_like_strip, SZ_METHOD_FLAGS, doc_strip},

    // Bidirectional operations
    {"find", (PyCFunction)Str_like_find, SZ_METHOD_FLAGS, doc_find},
    {"index", (PyCFunction)Str_like_index, SZ_METHOD_FLAGS, doc_index},
    {"partition", (PyCFunction)Str_like_partition, SZ_METHOD_FLAGS, doc_partition},
    {"split", (PyCFunction)Str_like_split, SZ_METHOD_FLAGS, doc_split},
    {"rfind", (PyCFunction)Str_like_rfind, SZ_METHOD_FLAGS, doc_rfind},
    {"rindex", (PyCFunction)Str_like_rindex, SZ_METHOD_FLAGS, doc_rindex},
    {"rpartition", (PyCFunction)Str_like_rpartition, SZ_METHOD_FLAGS, doc_rpartition},
    {"rsplit", (PyCFunction)Str_like_rsplit, SZ_METHOD_FLAGS, doc_rsplit},

    // Character search extensions
    {"find_first_of", (PyCFunction)Str_like_find_first_of, SZ_METHOD_FLAGS, doc_find_first_of},
    {"find_last_of", (PyCFunction)Str_like_find_last_of, SZ_METHOD_FLAGS, doc_find_last_of},
    {"find_first_not_of", (PyCFunction)Str_like_find_first_not_of, SZ_METHOD_FLAGS, doc_find_first_not_of},
    {"find_last_not_of", (PyCFunction)Str_like_find_last_not_of, SZ_METHOD_FLAGS, doc_find_last_not_of},
    {"count_byteset", (PyCFunction)Str_like_count_byteset, SZ_METHOD_FLAGS, doc_count_byteset},
    {"split_byteset", (PyCFunction)Str_like_split_byteset, SZ_METHOD_FLAGS, doc_split_byteset},
    {"rsplit_byteset", (PyCFunction)Str_like_rsplit_byteset, SZ_METHOD_FLAGS, doc_rsplit_byteset},

    // Lazily evaluated iterators
    {"split_iter", (PyCFunction)Str_like_split_iter, SZ_METHOD_FLAGS, doc_split_iter},
    {"rsplit_iter", (PyCFunction)Str_like_rsplit_iter, SZ_METHOD_FLAGS, doc_rsplit_iter},
    {"split_byteset_iter", (PyCFunction)Str_like_split_byteset_iter, SZ_METHOD_FLAGS, doc_split_byteset_iter},
    {"rsplit_byteset_iter", (PyCFunction)Str_like_rsplit_byteset_iter, SZ_METHOD_FLAGS, doc_rsplit_byteset_iter},

    // UTF-8 aware operations
    {"utf8_count", (PyCFunction)Str_like_utf8_count, SZ_METHOD_FLAGS, doc_utf8_count},
    {"utf8_split_newlines", (PyCFunction)Str_like_utf8_split_newlines, SZ_METHOD_FLAGS, doc_utf8_split_newlines},
    {"utf8_newlines", (PyCFunction)Str_like_utf8_newlines, SZ_METHOD_FLAGS, doc_utf8_newlines},
    {"utf8_split_whitespaces", (PyCFunction)Str_like_utf8_split_whitespaces, SZ_METHOD_FLAGS,
     doc_utf8_split_whitespaces},
    {"utf8_whitespaces", (PyCFunction)Str_like_utf8_whitespaces, SZ_METHOD_FLAGS, doc_utf8_whitespaces},
    {"utf8_split_delimiters", (PyCFunction)Str_like_utf8_split_delimiters, SZ_METHOD_FLAGS, doc_utf8_split_delimiters},
    {"utf8_delimiters", (PyCFunction)Str_like_utf8_delimiters, SZ_METHOD_FLAGS, doc_utf8_delimiters},
    {"utf8_wordbreaks", (PyCFunction)Str_like_utf8_wordbreaks, SZ_METHOD_FLAGS, doc_utf8_wordbreaks},
    {"utf8_codepoints", (PyCFunction)Str_like_utf8_codepoints, SZ_METHOD_FLAGS, doc_utf8_codepoints},
    {"utf8_graphemes", (PyCFunction)Str_like_utf8_graphemes, SZ_METHOD_FLAGS, doc_utf8_graphemes},
    {"utf8_sentences", (PyCFunction)Str_like_utf8_sentences, SZ_METHOD_FLAGS, doc_utf8_sentences},
    {"utf8_linebreaks", (PyCFunction)Str_like_utf8_linebreaks, SZ_METHOD_FLAGS, doc_utf8_linebreaks},
    {"utf8_uncased_fold", (PyCFunction)Str_like_utf8_uncased_fold, SZ_METHOD_FLAGS, doc_utf8_uncased_fold},
    {"utf8_norm", (PyCFunction)Str_like_utf8_norm, SZ_METHOD_FLAGS, doc_utf8_norm},
    {"utf8_find_denormalized", (PyCFunction)Str_like_utf8_find_denormalized, SZ_METHOD_FLAGS,
     doc_utf8_find_denormalized},
    {"utf8_uncased_search", (PyCFunction)Str_like_utf8_uncased_search, SZ_METHOD_FLAGS, doc_utf8_uncased_search},
    {"utf8_uncased_matches", (PyCFunction)Str_like_utf8_uncased_matches, SZ_METHOD_FLAGS, doc_utf8_uncased_matches},
    {"utf8_uncased_order", (PyCFunction)Str_like_utf8_uncased_order, SZ_METHOD_FLAGS, doc_utf8_uncased_order},

    // Dealing with larger-than-memory datasets
    {"offset_within", (PyCFunction)Str_offset_within, SZ_METHOD_FLAGS, doc_offset_within},
    {"write_to", (PyCFunction)Str_write_to, SZ_METHOD_FLAGS, doc_write_to},

    // In-place transforms
    {"translate", (PyCFunction)Str_like_translate, SZ_METHOD_FLAGS, doc_translate},
    {"fill_random", (PyCFunction)Str_like_fill_random, SZ_METHOD_FLAGS, doc_fill_random},

    {NULL, NULL, 0, NULL} // Sentinel
};

static char const doc_Str[] =                                                                //
    "Str(source)\n"                                                                          //
    "\n"                                                                                     //
    "Immutable byte-string/slice class with SIMD and SWAR-accelerated operations.\n"         //
    "Provides high-performance byte-string operations using modern CPU instructions.\n"      //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  source (str, bytes, bytearray, or buffer): Source data to wrap.\n"                    //
    "\n"                                                                                     //
    "Methods:\n"                                                                             //
    "  - find(), rfind(): Fast substring search with SIMD acceleration\n"                    //
    "  - count(): Count occurrences with optional overlap support\n"                         //
    "  - split(), rsplit(): String splitting with various separators\n"                      //
    "  - contains(): Fast membership testing\n"                                              //
    "  - translate(): Byte-level translations with lookup tables\n"                          //
    "\n"                                                                                     //
    "Example:\n"                                                                             //
    "  >>> text = sz.Str('the quick brown fox')\n"                                           //
    "  >>> text.find('brown')        # SIMD-accelerated substring search\n"                  //
    "  10\n"                                                                                 //
    "  >>> str(text[10:15])          # slicing returns a zero-copy view, not a new buffer\n" //
    "  'brown'";

PyTypeObject StrType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "stringzilla.Str",
    .tp_doc = doc_Str,
    .tp_basicsize = sizeof(Str),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Str_new,
    .tp_init = Str_init,
    .tp_dealloc = Str_dealloc,
    .tp_hash = Str_hash,
    .tp_richcompare = Str_richcompare,
    .tp_repr = (reprfunc)Str_repr,
    .tp_str = Str_str,
    .tp_methods = Str_methods,
    .tp_as_sequence = &Str_as_sequence,
    .tp_as_mapping = &Str_as_mapping,
    .tp_as_buffer = &Str_as_buffer,
    .tp_as_number = &Str_as_number,
    .tp_getset = Str_getsetters,
};
