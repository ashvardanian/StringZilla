/**
 *  @brief  Very light-weight CPython wrapper for StringZilla, with support for memory-mapping,
 *          native Python strings, Apache Arrow collections, and more.
 */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define NOMINMAX
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/stat.h> // `stat`
#include <sys/mman.h> // `mmap`
#include <fcntl.h>    // `O_RDNLY`
#endif

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <unistd.h> // `ssize_t`
#endif

#include <Python.h>

#include <stringzilla.h>

#pragma region Forward Declarations

static PyTypeObject FileType;
static PyTypeObject StrType;
static PyTypeObject StrsType;

struct {
    void *start;
    size_t length;
} temporary_memory = {NULL, 0};

/**
 *  @brief  Describes an on-disk file mapped into RAM, which is different from Python's
 *          native `mmap` module, as it exposes the address of the mapping in memory.
 */
typedef struct {
    PyObject_HEAD;
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    HANDLE file_handle;
    HANDLE mapping_handle;
#else
    int file_descriptor;
#endif
    void *start;
    size_t length;
} File;

/**
 *  @brief  Type-punned StringZilla-string, that points to a slice of an existing Python `str`
 *          or a `File`.
 *
 *  When a slice is constructed, the `parent` object's reference count is being incremented to preserve lifetime.
 *  It usage in Python would look like:
 *
 *      - Str() # Empty string
 *      - Str("some-string") # Full-range slice of a Python `str`
 *      - Str(File("some-path.txt")) # Full-range view of a persisted file
 *      - Str(File("some-path.txt"), from=0, to=sys.maxint)
 */
typedef struct {
    PyObject_HEAD;
    PyObject *parent;
    char const *start;
    size_t length;
} Str;

/**
 *  @brief  Variable length Python object similar to `Tuple[Union[Str, str]]`,
 *          for faster sorting, shuffling, joins, and lookups.
 */
typedef struct {
    PyObject_HEAD;

    enum {
        STRS_CONSECUTIVE_32,
        STRS_CONSECUTIVE_64,
        STRS_REORDERED,
        STRS_MULTI_SOURCE,
    } type;

    union {
        /**
         *  Simple structure resembling Apache Arrow arrays of variable length strings.
         *  When you split a `Str`, that is under 4 GB in size, this is used for space-efficiency.
         *  The `end_offsets` contains `count`-many integers marking the end offset of part at a given
         *  index. The length of consecutive elements can be determined as the difference in consecutive
         *  offsets. The starting offset of the first element is zero bytes after the `start`.
         *  Every chunk will include a separator of length `separator_length` at the end, except for the
         *  last one.
         */
        struct consecutive_slices_32bit_t {
            size_t count;
            size_t separator_length;
            PyObject *parent;
            char const *start;
            uint32_t *end_offsets;
        } consecutive_32bit;

        /**
         *  Simple structure resembling Apache Arrow arrays of variable length strings.
         *  When you split a `Str`, over 4 GB long, this structure is used to indicate chunk offsets.
         *  The `end_offsets` contains `count`-many integers marking the end offset of part at a given
         *  index. The length of consecutive elements can be determined as the difference in consecutive
         *  offsets. The starting offset of the first element is zero bytes after the `start`.
         *  Every chunk will include a separator of length `separator_length` at the end, except for the
         *  last one.
         */
        struct consecutive_slices_64bit_t {
            size_t count;
            size_t separator_length;
            PyObject *parent;
            char const *start;
            uint64_t *end_offsets;
        } consecutive_64bit;

        /**
         *  Once you sort, shuffle, or reorganize slices making up a larger string, this structure
         *  cn be used for space-efficient lookups.
         */
        struct reordered_slices_t {
            size_t count;
            PyObject *parent;
            sz_haystack_t *parts;
        } reordered;

        /**
         *  Complex structure with two variable length chunks inside - for the parents and their slices.
         *  The parents are sorted in ascending order of their memory ranges, to let us rapidly locate the source
         *  with a binary search. The slices are preserved
         */
        struct multi_source_strings_t {
            size_t count;
            size_t parents_count;

            PyObject **parents;
            sz_haystack_t *parts;
        } multi_source;
    } data;

} Strs;

#pragma endregion

#pragma region Helpers

void slice(size_t length, ssize_t start, ssize_t end, size_t *normalized_offset, size_t *normalized_length) {

    // clang-format off
    // Normalize negative indices
    if (start < 0) start += length;
    if (end < 0) end += length;

    // Clamp indices to a valid range
    if (start < 0) start = 0;
    if (end < 0) end = 0;
    if (start > (ssize_t)length) start = length;
    if (end > (ssize_t)length) end = length;

    // Ensure start <= end
    if (start > end) start = end;
    // clang-format on

    *normalized_offset = start;
    *normalized_length = end - start;
}

int export_string_like(PyObject *object, char const **start, size_t *length) {
    if (PyUnicode_Check(object)) {
        // Handle Python str
        Py_ssize_t signed_length;
        *start = PyUnicode_AsUTF8AndSize(object, &signed_length);
        *length = (size_t)signed_length;
        return 1;
    }
    else if (PyBytes_Check(object)) {
        // Handle Python str
        Py_ssize_t signed_length;
        if (PyBytes_AsStringAndSize(object, (char **)start, &signed_length) == -1) {
            PyErr_SetString(PyExc_TypeError, "Mapping bytes failed");
            return 0;
        }
        *length = (size_t)signed_length;
        return 1;
    }
    else if (PyObject_TypeCheck(object, &StrType)) {
        Str *str = (Str *)object;
        *start = str->start;
        *length = str->length;
        return 1;
    }
    else if (PyObject_TypeCheck(object, &FileType)) {
        File *file = (File *)object;
        *start = file->start;
        *length = file->length;
        return 1;
    }
    return 0;
}

#pragma endregion

#pragma region Global Functions

static Py_ssize_t str_find_vectorcall_(PyObject *_, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    // Initialize with default values or positional arguments
    PyObject *haystack_obj = args[0];
    PyObject *needle_obj = args[1];
    PyObject *start_obj = (nargs > 2) ? args[2] : NULL;
    PyObject *end_obj = (nargs > 3) ? args[3] : NULL;

    // Parse keyword arguments to overwrite positional ones
    if (kwnames != NULL) {
        for (Py_ssize_t i = 0; i < PyTuple_Size(kwnames); ++i) {
            PyObject *key = PyTuple_GetItem(kwnames, i);
            PyObject *value = args[nargs + i];
            if (PyUnicode_CompareWithASCIIString(key, "start") == 0)
                start_obj = value;
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0)
                end_obj = value;
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return NULL;
            }
            if (PyErr_Occurred())
                return NULL;
        }
    }

    struct sz_haystack_t haystack;
    struct sz_needle_t needle;
    Py_ssize_t start, end;

    // Validate and convert `haystack` and `needle`
    needle.anomaly_offset = 0;
    if (!export_string_like(haystack_obj, &haystack.start, &haystack.length) ||
        !export_string_like(needle_obj, &needle.start, &needle.length)) {
        PyErr_SetString(PyExc_TypeError, "Haystack and needle must be string-like");
        return NULL;
    }

    // Validate and convert `start`
    if (start_obj) {
        start = PyLong_AsSsize_t(start_obj);
        if (start == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The start argument must be an integer");
            return NULL;
        }
    }
    else {
        start = 0;
    }

    // Validate and convert `end`
    if (end_obj) {
        end = PyLong_AsSsize_t(end_obj);
        if (end == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The end argument must be an integer");
            return NULL;
        }
    }
    else {
        end = PY_SSIZE_T_MAX;
    }

    // Limit the `haystack` range
    size_t normalized_offset, normalized_length;
    slice(haystack.length, start, end, &normalized_offset, &normalized_length);
    haystack.start += normalized_offset;
    haystack.length = normalized_length;

    // Perform contains operation
    size_t offset = sz_neon_find_substr(haystack, needle);
    if (offset == haystack.length)
        return -1;
    return (Py_ssize_t)offset;
}

static PyObject *str_find_vectorcall(PyObject *_, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    Py_ssize_t signed_offset = str_find_vectorcall_(NULL, args, nargsf, kwnames);
    return PyLong_FromSsize_t(signed_offset);
}

static PyObject *str_contains_vectorcall(PyObject *_, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    Py_ssize_t signed_offset = str_find_vectorcall_(NULL, args, nargsf, kwnames);
    if (signed_offset == -1) {
        Py_RETURN_FALSE;
    }
    else {
        Py_RETURN_TRUE;
    }
}

static PyObject *str_count_vectorcall(PyObject *_, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    // Initialize with default values or positional arguments
    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *haystack_obj = args[0];
    PyObject *needle_obj = args[1];
    PyObject *start_obj = (nargs > 2) ? args[2] : NULL;
    PyObject *end_obj = (nargs > 3) ? args[3] : NULL;
    PyObject *allowoverlap_obj = (nargs > 4) ? args[4] : NULL;

    // Parse keyword arguments to overwrite positional ones
    if (kwnames != NULL) {
        for (Py_ssize_t i = 0; i < PyTuple_Size(kwnames); ++i) {
            PyObject *key = PyTuple_GetItem(kwnames, i);
            PyObject *value = args[nargs + i];
            if (PyUnicode_CompareWithASCIIString(key, "start") == 0)
                start_obj = value;
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0)
                end_obj = value;
            else if (PyUnicode_CompareWithASCIIString(key, "allowoverlap") == 0)
                allowoverlap_obj = value;
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return NULL;
            }
            if (PyErr_Occurred())
                return NULL;
        }
    }

    struct sz_haystack_t haystack;
    struct sz_needle_t needle;
    int allowoverlap;
    Py_ssize_t start, end;

    // Validate and convert `haystack` and `needle`
    needle.anomaly_offset = 0;
    if (!export_string_like(haystack_obj, &haystack.start, &haystack.length) ||
        !export_string_like(needle_obj, &needle.start, &needle.length)) {
        PyErr_SetString(PyExc_TypeError, "Haystack and needle must be string-like");
        return NULL;
    }

    // Validate and convert `start`
    if (start_obj) {
        start = PyLong_AsSsize_t(start_obj);
        if (start == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The start argument must be an integer");
            return NULL;
        }
    }
    else {
        start = 0;
    }

    // Validate and convert `end`
    if (end_obj) {
        end = PyLong_AsSsize_t(end_obj);
        if (end == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The end argument must be an integer");
            return NULL;
        }
    }
    else {
        end = PY_SSIZE_T_MAX;
    }

    // Validate and convert `allowoverlap`
    if (allowoverlap_obj) {
        allowoverlap = PyObject_IsTrue(allowoverlap_obj);
        if (allowoverlap == -1) {
            PyErr_SetString(PyExc_TypeError, "The allowoverlap argument must be a boolean");
            return NULL;
        }
    }
    else {
        allowoverlap = 0;
    }

    // Limit the haystack range
    size_t normalized_offset, normalized_length;
    slice(haystack.length, start, end, &normalized_offset, &normalized_length);
    haystack.start += normalized_offset;
    haystack.length = normalized_length;

    // Perform counting operation
    size_t count = 0;
    if (needle.length == 1) {
        count = sz_naive_count_char(haystack, *needle.start);
    }
    else {
        // Your existing logic for count_substr can be embedded here
        if (allowoverlap) {
            while (haystack.length) {
                size_t offset = sz_neon_find_substr(haystack, needle);
                int found = offset != haystack.length;
                count += found;
                haystack.start += offset + found;
                haystack.length -= offset + found;
            }
        }
        else {
            while (haystack.length) {
                size_t offset = sz_neon_find_substr(haystack, needle);
                int found = offset != haystack.length;
                count += found;
                haystack.start += offset + needle.length;
                haystack.length -= offset + needle.length * found;
            }
        }
    }

    return PyLong_FromSize_t(count);
}

static PyObject *str_levenstein_vectorcall(PyObject *_, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    // Validate the number of arguments
    if (nargs < 2 || nargs > 3) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str1_obj = args[0];
    PyObject *str2_obj = args[1];

    struct sz_haystack_t str1, str2;
    if (!export_string_like(str1_obj, &str1.start, &str1.length) ||
        !export_string_like(str2_obj, &str2.start, &str2.length)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be string-like");
        return NULL;
    }

    // Initialize bound argument
    int bound = 255;

    // Check if `bound` is given as a positional argument
    if (nargs == 3) {
        bound = PyLong_AsLong(args[2]);
        if (bound > 255 || bound < 0) {
            PyErr_SetString(PyExc_ValueError, "Bound must be an integer between 0 and 255");
            return NULL;
        }
    }

    // Parse keyword arguments
    if (kwnames != NULL) {
        for (Py_ssize_t i = 0; i < PyTuple_Size(kwnames); ++i) {
            PyObject *key = PyTuple_GetItem(kwnames, i);
            PyObject *value = args[nargs + i];
            if (PyUnicode_CompareWithASCIIString(key, "bound") == 0) {
                if (nargs == 3) {
                    PyErr_SetString(PyExc_TypeError, "Received bound both as positional and keyword argument");
                    return NULL;
                }
                bound = PyLong_AsLong(value);
                if (bound > 255 || bound < 0) {
                    PyErr_SetString(PyExc_ValueError, "Bound must be an integer between 0 and 255");
                    return NULL;
                }
            }
        }
    }

    // Initialize or reallocate the Levenshtein distance matrix
    size_t memory_needed = sz_levenstein_memory_needed(str1.length, str2.length);
    if (temporary_memory.length < memory_needed) {
        temporary_memory.start = realloc(temporary_memory.start, memory_needed);
        temporary_memory.length = memory_needed;
    }
    if (temporary_memory.start == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for the Levenshtein matrix");
        return NULL;
    }

    levenstein_distance_t distance = sz_levenstein( //
        str1.start,
        str1.length,
        str2.start,
        str2.length,
        (levenstein_distance_t)bound,
        temporary_memory.start);
    return PyLong_FromLong(distance);
}

static PyObject *strs_split_vectorcall(PyObject *_, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    if (nargs < 1) {
        PyErr_SetString(PyExc_TypeError, "sz.split() requires at least 1 argument");
        return NULL;
    }

    // Initialize with default values or positional arguments
    PyObject *text_obj = args[0];
    PyObject *separator_obj = (nargs > 1) ? args[1] : NULL;
    PyObject *maxsplit_obj = (nargs > 2) ? args[2] : NULL;
    PyObject *keepseparator_obj = (nargs > 3) ? args[3] : NULL;

    // Parse keyword arguments to overwrite positional ones
    if (kwnames != NULL) {
        for (Py_ssize_t i = 0; i < PyTuple_Size(kwnames); ++i) {
            PyObject *key = PyTuple_GetItem(kwnames, i);
            PyObject *value = args[nargs + i];

            if (PyUnicode_CompareWithASCIIString(key, "separator") == 0)
                separator_obj = value;
            else if (PyUnicode_CompareWithASCIIString(key, "maxsplit") == 0)
                maxsplit_obj = value;
            else if (PyUnicode_CompareWithASCIIString(key, "keepseparator") == 0)
                keepseparator_obj = value;
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return NULL;
            }

            // Check for errors during conversion
            if (PyErr_Occurred())
                return NULL;
        }
    }

    struct sz_haystack_t text;
    struct sz_needle_t separator;
    int keepseparator;
    Py_ssize_t maxsplit;
    separator.anomaly_offset = 0;

    // Validate and convert `text`
    if (!export_string_like(text_obj, &text.start, &text.length)) {
        PyErr_SetString(PyExc_TypeError, "The text argument must be string-like");
        return NULL;
    }

    // Validate and convert `separator`
    if (separator_obj) {
        Py_ssize_t len;
        if (!export_string_like(separator_obj, &separator.start, &len)) {
            PyErr_SetString(PyExc_TypeError, "The separator argument must be string-like");
            return NULL;
        }
        separator.length = (size_t)len;
    }
    else {
        separator.start = " ";
        separator.length = 1;
    }

    // Validate and convert `keepseparator`
    if (keepseparator_obj) {
        keepseparator = PyObject_IsTrue(keepseparator_obj);
        if (keepseparator == -1) {
            PyErr_SetString(PyExc_TypeError, "The keepseparator argument must be a boolean");
            return NULL;
        }
    }
    else {
        keepseparator = 0;
    }

    // Validate and convert `maxsplit`
    if (maxsplit_obj) {
        maxsplit = PyLong_AsSsize_t(maxsplit_obj);
        if (maxsplit == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The maxsplit argument must be an integer");
            return NULL;
        }
    }
    else {
        maxsplit = PY_SSIZE_T_MAX;
    }

    // Create Strs object
    Strs *result = (Strs *)PyObject_New(Strs, &StrsType);
    if (!result)
        return NULL;

    // Initialize Strs object based on the splitting logic
    void *offsets_endings = NULL;
    size_t offsets_capacity = 0;
    size_t offsets_count = 0;
    size_t bytes_per_offset;
    if (text.length >= UINT32_MAX) {
        bytes_per_offset = 8;
        result->type = STRS_CONSECUTIVE_64;
        result->data.consecutive_64bit.start = text.start;
        result->data.consecutive_64bit.parent = text_obj;
        result->data.consecutive_64bit.separator_length = !keepseparator * separator.length;
    }
    else {
        bytes_per_offset = 4;
        result->type = STRS_CONSECUTIVE_32;
        result->data.consecutive_32bit.start = text.start;
        result->data.consecutive_32bit.parent = text_obj;
        result->data.consecutive_32bit.separator_length = !keepseparator * separator.length;
    }

    // Iterate through string, keeping track of the
    sz_size_t last_start = 0;
    while (last_start < text.length && offsets_count < maxsplit) {
        sz_haystack_t text_remaining;
        text_remaining.start = text.start + last_start;
        text_remaining.length = text.length - last_start;
        sz_size_t offset_in_remaining = sz_neon_find_substr(text_remaining, separator);

        // Reallocate offsets array if needed
        if (offsets_count >= offsets_capacity) {
            offsets_capacity = (offsets_capacity + 1) * 2;
            void *new_offsets = realloc(offsets_endings, offsets_capacity * bytes_per_offset);
            if (!new_offsets) {
                if (offsets_endings)
                    free(offsets_endings);
            }
            offsets_endings = new_offsets;
        }

        // If the memory allocation has failed - discard the response
        if (!offsets_endings) {
            Py_XDECREF(result);
            PyErr_NoMemory();
            return NULL;
        }

        // Export the offset
        size_t will_continue = offset_in_remaining != text_remaining.length;
        size_t next_offset = last_start + offset_in_remaining + separator.length * will_continue;
        if (text.length >= UINT32_MAX)
            ((uint64_t *)offsets_endings)[offsets_count++] = (uint64_t)next_offset;
        else
            ((uint32_t *)offsets_endings)[offsets_count++] = (uint32_t)next_offset;

        // Next time we want to start
        last_start = last_start + offset_in_remaining + separator.length;
    }

    // Populate the Strs object with the offsets
    if (text.length >= UINT32_MAX) {
        result->data.consecutive_64bit.end_offsets = offsets_endings;
        result->data.consecutive_64bit.count = offsets_count;
    }
    else {
        result->data.consecutive_32bit.end_offsets = offsets_endings;
        result->data.consecutive_32bit.count = offsets_count;
    }

    Py_INCREF(text_obj);
    return (PyObject *)result;
}

#pragma endregion

#pragma region MemoryMappingFile

static void File_dealloc(File *self) {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    if (self->start) {
        UnmapViewOfFile(self->start);
        self->start = NULL;
    }
    if (self->mapping_handle) {
        CloseHandle(self->mapping_handle);
        self->mapping_handle = NULL;
    }
    if (self->file_handle) {
        CloseHandle(self->file_handle);
        self->file_handle = NULL;
    }
#else
    if (self->start) {
        munmap(self->start, self->length);
        self->start = NULL;
        self->length = 0;
    }
    if (self->file_descriptor != 0) {
        close(self->file_descriptor);
        self->file_descriptor = 0;
    }
#endif
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *File_new(PyTypeObject *type, PyObject *positional_args, PyObject *named_args) {
    File *self;
    self = (File *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    self->file_handle = NULL;
    self->mapping_handle = NULL;
#else
    self->file_descriptor = 0;
#endif
    self->start = NULL;
    self->length = 0;
}

static int File_init(File *self, PyObject *positional_args, PyObject *named_args) {
    const char *path;
    if (!PyArg_ParseTuple(positional_args, "s", &path))
        return -1;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    self->file_handle = CreateFile(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (self->file_handle == INVALID_HANDLE_VALUE) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't map the file!");
        return -1;
    }

    self->mapping_handle = CreateFileMapping(self->file_handle, 0, PAGE_READONLY, 0, 0, 0);
    if (self->mapping_handle == 0) {
        CloseHandle(self->file_handle);
        self->file_handle = NULL;
        PyErr_SetString(PyExc_RuntimeError, "Couldn't map the file!");
        return -1;
    }

    char *file = (char *)MapViewOfFile(self->mapping_handle, FILE_MAP_READ, 0, 0, 0);
    if (file == 0) {
        CloseHandle(self->mapping_handle);
        self->mapping_handle = NULL;
        CloseHandle(self->file_handle);
        self->file_handle = NULL;
        PyErr_SetString(PyExc_RuntimeError, "Couldn't map the file!");
        return -1;
    }
    self->start = file;
    self->length = GetFileSize(self->file_handle, 0);
#else
    struct stat sb;
    self->file_descriptor = open(path, O_RDONLY);
    if (fstat(self->file_descriptor, &sb) != 0) {
        close(self->file_descriptor);
        self->file_descriptor = 0;
        PyErr_SetString(PyExc_RuntimeError, "Can't retrieve file size!");
        return -1;
    }
    size_t file_size = sb.st_size;
    void *map = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, self->file_descriptor, 0);
    if (map == MAP_FAILED) {
        close(self->file_descriptor);
        self->file_descriptor = 0;
        PyErr_SetString(PyExc_RuntimeError, "Couldn't map the file!");
        return -1;
    }
    self->start = map;
    self->length = file_size;
#endif

    return 0;
}

static PyMethodDef File_methods[] = { //
    {NULL, NULL, 0, NULL}};

static PyTypeObject FileType = {
    PyObject_HEAD_INIT(NULL).tp_name = "stringzilla.File",
    .tp_doc = "Memory mapped file class, that exposes the memory range for low-level access",
    .tp_basicsize = sizeof(File),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = File_methods,
    .tp_new = (newfunc)File_new,
    .tp_init = (initproc)File_init,
    .tp_dealloc = (destructor)File_dealloc,

    // PyBufferProcs *tp_as_buffer;

    // reprfunc tp_repr;
    // PyNumberMethods *tp_as_number;
    // PySequenceMethods *tp_as_sequence;
    // PyMappingMethods *tp_as_mapping;
    // ternaryfunc tp_call;
    // reprfunc tp_str;
    // getattrofunc tp_getattro;
    // setattrofunc tp_setattro;
};

#pragma endregion

#pragma region Str

static int Str_init(Str *self, PyObject *positional_args, PyObject *named_args) {
    PyObject *parent = NULL;
    Py_ssize_t from = 0;
    Py_ssize_t to = PY_SSIZE_T_MAX;

    // The `named_args` would be `NULL`
    if (named_args) {
        static char *names[] = {"parent", "from", "to", NULL};
        if (!PyArg_ParseTupleAndKeywords(positional_args, named_args, "|Onn", names, &parent, &from, &to))
            return -1;
    }
    else if (!PyArg_ParseTuple(positional_args, "|Onn", &parent, &from, &to))
        return -1;

    // Handle empty string
    if (parent == NULL) {
        self->start = NULL;
        self->length = 0;
    }
    // Increment the reference count of the parent
    else if (export_string_like(parent, &self->start, &self->length)) {
        self->parent = parent;
        Py_INCREF(parent);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Unsupported parent type");
        return -1;
    }

    // Apply slicing
    size_t normalized_offset, normalized_length;
    slice(self->length, from, to, &normalized_offset, &normalized_length);
    self->start = ((char *)self->start) + normalized_offset;
    self->length = normalized_length;
    return 0;
}

static PyObject *Str_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Str *self;
    self = (Str *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;

    self->parent = NULL;
    self->start = NULL;
    self->length = 0;
    return (PyObject *)self;
}

static void Str_dealloc(Str *self) {
    if (self->parent)
        Py_XDECREF(self->parent);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Str_str(Str *self) { return PyUnicode_FromStringAndSize(self->start, self->length); }

static Py_hash_t Str_hash(Str *self) { return (Py_hash_t)sz_hash_crc32_native(self->start, self->length); }

static Py_ssize_t Str_len(Str *self) { return self->length; }

static PyObject *Str_getitem(Str *self, Py_ssize_t i) {

    // Negative indexing
    if (i < 0)
        i += self->length;

    if (i < 0 || (size_t)i >= self->length) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }

    // Assuming the underlying data is UTF-8 encoded
    return PyUnicode_FromStringAndSize(self->start + i, 1);
}

static PyObject *Str_subscript(Str *self, PyObject *key) {
    if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step;
        if (PySlice_Unpack(key, &start, &stop, &step) < 0)
            return NULL;
        if (PySlice_AdjustIndices(self->length, &start, &stop, step) < 0)
            return NULL;

        if (step != 1) {
            PyErr_SetString(PyExc_IndexError, "Efficient step is not supported");
            return NULL;
        }

        // Create a new `Str` object
        Str *self_slice = (Str *)StrType.tp_alloc(&StrType, 0);
        if (self_slice == NULL && PyErr_NoMemory())
            return NULL;

        // Set its properties based on the slice
        self_slice->start = self->start + start;
        self_slice->length = stop - start;
        self_slice->parent = (PyObject *)self; // Set parent to keep it alive

        // Increment the reference count of the parent
        Py_INCREF(self);
        return (PyObject *)self_slice;
    }
    else if (PyLong_Check(key)) {
        return Str_getitem(self, PyLong_AsSsize_t(key));
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Str indices must be integers or slices");
        return NULL;
    }
}

// Will be called by the `PySequence_Contains`
static int Str_contains(Str *self, PyObject *arg) {

    struct sz_needle_t needle_struct;
    needle_struct.anomaly_offset = 0;
    if (!export_string_like(arg, &needle_struct.start, &needle_struct.length)) {
        PyErr_SetString(PyExc_TypeError, "Unsupported argument type");
        return -1;
    }

    struct sz_haystack_t haystack;
    haystack.start = self->start;
    haystack.length = self->length;
    size_t position = sz_neon_find_substr(haystack, needle_struct);
    return position != haystack.length;
}

static Py_ssize_t Strs_len(Strs *self) {
    switch (self->type) {
    case STRS_CONSECUTIVE_32: return self->data.consecutive_32bit.count;
    case STRS_CONSECUTIVE_64: return self->data.consecutive_64bit.count;
    case STRS_REORDERED: return self->data.reordered.count;
    case STRS_MULTI_SOURCE: return self->data.multi_source.count;
    default: return 0;
    }
}

static PyObject *Strs_getitem(Strs *self, Py_ssize_t i) {
    // Check for negative index and convert to positive
    Py_ssize_t count = Strs_len(self);
    if (i < 0)
        i += count;
    if (i < 0 || i >= count) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }

    PyObject *parent = NULL;
    char const *start = NULL;
    size_t length = 0;

    // Extract a member element based on
    switch (self->type) {
    case STRS_CONSECUTIVE_32: {
        uint32_t start_offset = (i == 0) ? 0 : self->data.consecutive_32bit.end_offsets[i - 1];
        uint32_t end_offset = self->data.consecutive_32bit.end_offsets[i];
        start = self->data.consecutive_32bit.start + start_offset;
        length = end_offset - start_offset - self->data.consecutive_32bit.separator_length * (i + 1 != count);
        parent = self->data.consecutive_32bit.parent;
        break;
    }
    case STRS_CONSECUTIVE_64: {
        uint64_t start_offset = (i == 0) ? 0 : self->data.consecutive_64bit.end_offsets[i - 1];
        uint64_t end_offset = self->data.consecutive_64bit.end_offsets[i];
        start = self->data.consecutive_64bit.start + start_offset;
        length = end_offset - start_offset - self->data.consecutive_64bit.separator_length * (i + 1 != count);
        parent = self->data.consecutive_64bit.parent;
        break;
    }
    case STRS_REORDERED: {
        //
        break;
    }
    case STRS_MULTI_SOURCE: {
        //
        break;
    }
    default: PyErr_SetString(PyExc_TypeError, "Unknown Strs kind"); return NULL;
    }

    // Create a new `Str` object
    Str *parent_slice = (Str *)StrType.tp_alloc(&StrType, 0);
    if (parent_slice == NULL && PyErr_NoMemory())
        return NULL;

    parent_slice->start = start;
    parent_slice->length = length;
    parent_slice->parent = parent;
    Py_INCREF(parent);
    return parent_slice;
}

static PyObject *Strs_subscript(Str *self, PyObject *key) {
    if (PyLong_Check(key))
        return Strs_getitem(self, PyLong_AsSsize_t(key));
    return NULL;
}

// Will be called by the `PySequence_Contains`
static int Strs_contains(Str *self, PyObject *arg) { return 0; }

static PyObject *Str_richcompare(PyObject *self, PyObject *other, int op) {

    char const *a_start, *b_start;
    size_t a_length, b_length;
    if (!export_string_like(self, &a_start, &a_length) || !export_string_like(other, &b_start, &b_length))
        Py_RETURN_NOTIMPLEMENTED;

    // Perform byte-wise comparison up to the minimum length
    size_t min_length = a_length < b_length ? a_length : b_length;
    int cmp_result = memcmp(a_start, b_start, min_length);

    // If the strings are equal up to `min_length`, then the shorter string is smaller
    if (cmp_result == 0)
        cmp_result = (a_length > b_length) - (a_length < b_length);

    switch (op) {
    case Py_LT: return PyBool_FromLong(cmp_result < 0);
    case Py_LE: return PyBool_FromLong(cmp_result <= 0);
    case Py_EQ: return PyBool_FromLong(cmp_result == 0);
    case Py_NE: return PyBool_FromLong(cmp_result != 0);
    case Py_GT: return PyBool_FromLong(cmp_result > 0);
    case Py_GE: return PyBool_FromLong(cmp_result >= 0);
    default: Py_RETURN_NOTIMPLEMENTED;
    }
}

static PySequenceMethods Str_as_sequence = {
    .sq_length = Str_len,        //
    .sq_item = Str_getitem,      //
    .sq_contains = Str_contains, //
};

static PyMappingMethods Str_as_mapping = {
    .mp_length = Str_len,          //
    .mp_subscript = Str_subscript, // Is used to implement slices in Python
};

static PyMethodDef Str_methods[] = {
    // {"contains", (PyCFunction)..., METH_NOARGS, "Convert to Python `str`"},
    // {"find", (PyCFunction)..., METH_NOARGS, "Get length"},
    // {"__getitem__", (PyCFunction)..., METH_O, "Indexing"},
    {NULL, NULL, 0, NULL}};

static PyTypeObject StrType = {
    PyObject_HEAD_INIT(NULL).tp_name = "stringzilla.Str",
    .tp_doc = "Immutable string/slice class with SIMD and SWAR-accelerated operations",
    .tp_basicsize = sizeof(Str),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = Str_methods,
    .tp_new = Str_new,
    .tp_init = Str_init,
    .tp_dealloc = Str_dealloc,
    .tp_as_sequence = &Str_as_sequence,
    .tp_as_mapping = &Str_as_mapping,
    .tp_hash = Str_hash, // String hashing functions
    .tp_richcompare = Str_richcompare,
    .tp_str = Str_str,
    // .tp_as_buffer = (PyBufferProcs *)NULL, // Functions to access object as input/output buffer
};

static PySequenceMethods Strs_as_sequence = {
    .sq_length = Strs_len,        //
    .sq_item = Strs_getitem,      //
    .sq_contains = Strs_contains, //
};

static PyMappingMethods Strs_as_mapping = {
    .mp_length = Strs_len,          //
    .mp_subscript = Strs_subscript, // Is used to implement slices in Python
};

static PyTypeObject StrsType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Strs",
    .tp_doc = "Space-efficient container for large collections of strings and their slices",
    .tp_basicsize = sizeof(Strs),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_as_sequence = &Strs_as_sequence,
    .tp_as_mapping = &Strs_as_mapping,
};

#pragma endregion

static PyMethodDef stringzilla_methods[] = { //
    {NULL, NULL, 0, NULL}};

static PyModuleDef stringzilla_module = {
    PyModuleDef_HEAD_INIT,
    "stringzilla",
    "Crunch 100+ GB Strings in Python with ease",
    -1,
    stringzilla_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyObject *register_vectorcall(PyObject *module, char const *name, vectorcallfunc vectorcall) {

    PyCFunctionObject *vectorcall_object = (PyCFunctionObject *)PyObject_Malloc(sizeof(PyCFunctionObject));
    if (vectorcall_object == NULL)
        return NULL;

    PyObject_Init(vectorcall_object, &PyCFunction_Type);
    vectorcall_object->m_ml = NULL; // No regular `PyMethodDef`
    vectorcall_object->vectorcall = vectorcall;

    // Add the 'find' function to the module
    if (PyModule_AddObject(module, name, vectorcall_object) < 0) {
        Py_XDECREF(vectorcall_object);
        return NULL;
    }
    return vectorcall_object;
}

void cleanup_module(void) {
    if (temporary_memory.start)
        free(temporary_memory.start);
    temporary_memory.start = NULL;
    temporary_memory.length = 0;
}

PyMODINIT_FUNC PyInit_stringzilla(void) {
    PyObject *m;

    if (PyType_Ready(&StrType) < 0)
        return NULL;

    if (PyType_Ready(&FileType) < 0)
        return NULL;

    if (PyType_Ready(&StrsType) < 0)
        return NULL;

    m = PyModule_Create(&stringzilla_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&StrType);
    if (PyModule_AddObject(m, "Str", (PyObject *)&StrType) < 0) {
        Py_XDECREF(&StrType);
        Py_XDECREF(m);
        return NULL;
    }

    Py_INCREF(&FileType);
    if (PyModule_AddObject(m, "File", (PyObject *)&FileType) < 0) {
        Py_XDECREF(&FileType);
        Py_XDECREF(&StrType);
        Py_XDECREF(m);
        return NULL;
    }

    Py_INCREF(&StrsType);
    if (PyModule_AddObject(m, "Strs", (PyObject *)&StrsType) < 0) {
        Py_XDECREF(&StrsType);
        Py_XDECREF(&FileType);
        Py_XDECREF(&StrType);
        Py_XDECREF(m);
        return NULL;
    }

    // Initialize temporary_memory, if needed
    temporary_memory.start = malloc(4096);
    temporary_memory.length = 4096 * (temporary_memory.start != NULL);
    // atexit(cleanup_module);

    // Register the vectorized functions
    PyObject *vectorized_find = register_vectorcall(m, "find", str_find_vectorcall);
    PyObject *vectorized_contains = register_vectorcall(m, "contains", str_contains_vectorcall);
    PyObject *vectorized_count = register_vectorcall(m, "count", str_count_vectorcall);
    PyObject *vectorized_levenstein = register_vectorcall(m, "levenstein", str_levenstein_vectorcall);

    PyObject *vectorized_split = register_vectorcall(m, "split", strs_split_vectorcall);
    PyObject *vectorized_sort = register_vectorcall(m, "sort", str_find_vectorcall);
    PyObject *vectorized_shuffle = register_vectorcall(m, "shuffle", str_find_vectorcall);

    if (!vectorized_find || !vectorized_count ||          //
        !vectorized_contains || !vectorized_levenstein || //
        !vectorized_split || !vectorized_sort || !vectorized_shuffle) {
        goto cleanup;
    }

    return m;

cleanup:
    if (vectorized_find)
        Py_XDECREF(vectorized_find);
    if (vectorized_contains)
        Py_XDECREF(vectorized_contains);
    if (vectorized_count)
        Py_XDECREF(vectorized_count);
    if (vectorized_levenstein)
        Py_XDECREF(vectorized_levenstein);
    if (vectorized_split)
        Py_XDECREF(vectorized_split);
    if (vectorized_sort)
        Py_XDECREF(vectorized_sort);
    if (vectorized_shuffle)
        Py_XDECREF(vectorized_shuffle);

    Py_XDECREF(&FileType);
    Py_XDECREF(&StrType);
    Py_XDECREF(m);
    return NULL;
}