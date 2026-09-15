/**
 *  @brief The module-init glue — `PyModuleDef`, type registration, and `PyInit_stringzilla`.
 *  @file python/stringzilla/stringzilla.c
 *  @author Ash Vardanian
 *
 *  - Doesn't use PyBind11, NanoBind, Boost.Python, or any other high-level libs, only CPython API.
 *  - To minimize latency this implementation avoids `PyArg_ParseTupleAndKeywords` calls.
 *  - Reimplements all of the `str` functionality in C as a `Str` type.
 *  - Provides a highly generic `Strs` class for handling collections of strings, Arrow-style or not.
 *
 *  Pandas doesn't provide a C API, and even in the 2.0 the Apache Arrow representation is opt-in, not default.
 *  PyCapsule protocol in conjunction with @b `__arrow_c_array__` dunder methods can be used to extract strings.
 *  @see https://arrow.apache.org/docs/python/generated/pyarrow.array.html
 *
 *  This module exports C functions via `PyCapsule` of `PyAPI` for use by other extensions (like `stringzillas-cpus`):
 *  - `sz_py_export_string_like`.
 *  - `sz_py_export_strings_as_sequence`.
 *  - `sz_py_export_strings_as_u32tape`.
 *  - `sz_py_export_strings_as_u64tape`.
 *  - `sz_py_replace_strings_allocator`.
 *
 *  Function Naming Convention:
 *  - `Str_like_*`: Functions that can be called both as module-level functions AND as member methods.
 *  - `Str_*`: Functions that are member-only methods or have simpler calling conventions.
 *
 *  This translation unit owns only the module-init glue: `PyModuleDef`, the type-registration table, and
 *  `PyInit_stringzilla`. The `File`/`Str`/`Strs` struct layouts and the `PyTypeObject` forward declarations
 *  every domain file needs live in `stringzilla.h`; the domains themselves are split across `shared.c`, `file.c`,
 *  `str.c`, `strs.c`, `memory.c`, `hash.c`, `cipher.c`, `find.c`, `compare.c`, `sort.c`, `intersect.c`,
 *  and the `utf8_*.c` files.
 */
#include "stringzilla.h"

/**
 *  @brief  The function table `stringzillas` reads out of this module's `_sz_py_api` capsule.
 *
 *  `python/stringzillas.c` carries its own copy of this layout and casts the capsule pointer to it. The two
 *  must stay identical: a field added, removed, or reordered on one side alone makes the other misread memory
 *  with no diagnostic, since the capsule carries no version tag.
 */
typedef struct PyAPI {
    sz_bool_t (*sz_py_export_string_like)(PyObject *, sz_cptr_t *, sz_size_t *);
    sz_bool_t (*sz_py_export_strings_as_sequence)(PyObject *, sz_sequence_t *);
    sz_bool_t (*sz_py_export_strings_as_u32tape)(PyObject *, sz_cptr_t *, sz_u32_t const **, sz_size_t *);
    sz_bool_t (*sz_py_export_strings_as_u64tape)(PyObject *, sz_cptr_t *, sz_u64_t const **, sz_size_t *);
    sz_bool_t (*sz_py_replace_strings_allocator)(PyObject *, sz_memory_allocator_t *);
} PyAPI;

/** @brief  Builds a tuple of the capability names in @p caps, or @c NULL with a Python exception set. */
static PyObject *capabilities_to_tuple(sz_capability_t caps) {
    sz_cptr_t cap_strings[SZ_CAPABILITIES_COUNT];
    sz_size_t cap_count = sz_capabilities_to_strings_implementation_(caps, cap_strings, SZ_CAPABILITIES_COUNT);
    PyObject *caps_tuple = PyTuple_New(cap_count);
    if (!caps_tuple) return NULL;
    for (sz_size_t i = 0; i < cap_count; i++) {
        PyObject *cap_str = PyUnicode_FromString(cap_strings[i]);
        if (!cap_str) {
            Py_DECREF(caps_tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(caps_tuple, i, cap_str);
    }
    return caps_tuple;
}

static int parse_and_intersect_capabilities(PyObject *caps_obj, sz_capability_t *result) {
    if (!caps_obj) {
        PyErr_SetString(PyExc_TypeError, "capabilities must be a tuple or list of strings");
        return -1;
    }
    // A tuple snapshot rather than `PySequence_Fast`, which hands back the caller's list itself and leaves
    // the walk indexing into storage another thread can resize.
    PyObject *seq = PySequence_Tuple(caps_obj);
    if (!seq) {
        PyErr_SetString(PyExc_TypeError, "capabilities must be a tuple or list of strings");
        return -1;
    }

    sz_capability_t requested_caps = 0;
    Py_ssize_t n = PyTuple_GET_SIZE(seq);

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyTuple_GET_ITEM(seq, i);
        if (!PyUnicode_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "capabilities must be strings");
            Py_DECREF(seq);
            return -1;
        }
        char const *cap_str = PyUnicode_AsUTF8(item);
        if (!cap_str) {
            Py_DECREF(seq);
            return -1;
        }
        sz_capability_t flag = sz_capability_from_string_implementation_(cap_str);
        if (flag == sz_caps_none_k) {
            PyErr_Format(PyExc_ValueError, "Unknown capability: %s", cap_str);
            Py_DECREF(seq);
            return -1;
        }
        requested_caps |= flag;
    }
    Py_DECREF(seq);

    // An empty request or one entirely outside this machine's reach is a configuration error; silently
    // degrading to serial would hide it.
    *result = requested_caps & sz_capabilities();
    if (*result == 0) {
        char available[256];
        sz_capabilities_to_string_implementation_(sz_capabilities(), available, sizeof(available));
        PyErr_Format(PyExc_ValueError, "No requested capability is available here; available: %s", available);
        return -1;
    }
    return 0;
}

static char const doc_reset_capabilities[] =                                         //
    "reset_capabilities(names) -> None\n\n"                                          //
    "Sets the active SIMD/backend capabilities for this module and updates the\n"    //
    "runtime dispatch table. The provided names are intersected with hardware\n"     //
    "capabilities; raises ValueError if none of them is available.\n\n"              //
    "Side effects: updates stringzilla.__capabilities__ and __capabilities_str__.\n" //
    "\n"                                                                             //
    "Example:\n"                                                                     //
    "  >>> sz.reset_capabilities(['serial'])  # restrict dispatch to the scalar backend";

static PyObject *module_reset_capabilities(PyObject *self, PyObject *args) {
    PyObject *caps_obj = NULL;
    if (!PyArg_ParseTuple(args, "O", &caps_obj)) return NULL;

    sz_capability_t caps = 0;
    if (parse_and_intersect_capabilities(caps_obj, &caps) != 0) return NULL;

    // Update the dispatch table
    sz_dispatch_table_update(caps);

    // Recompute and set module-level capability exports
    PyObject *caps_tuple = capabilities_to_tuple(caps);
    if (!caps_tuple) return NULL;
    if (PyObject_SetAttrString(self, "__capabilities__", caps_tuple) != 0) {
        Py_DECREF(caps_tuple);
        return NULL;
    }
    Py_DECREF(caps_tuple);

    sz_cptr_t caps_str = sz_capabilities_to_string(caps);
    if (PyObject_SetAttrString(self, "__capabilities_str__", PyUnicode_FromString(caps_str)) != 0) { return NULL; }

    Py_RETURN_NONE;
}

static void stringzilla_cleanup(PyObject *m) {
    // Drain both free-lists, releasing the headers parked for reuse during the interpreter's lifetime.
    stringzilla_state_t *state = (stringzilla_state_t *)PyModule_GetState(m);
    if (!state) return;
    for (Str *node = state->str_freelist_head; node;) {
        Str *next = (Str *)node->parent;
        PyObject_Free(node);
        node = next;
    }
    for (Strs *node = state->strs_freelist_head; node;) {
        Strs *next = *Strs_freelist_next_(node);
        PyObject_Free(node);
        node = next;
    }
    state->str_freelist_head = NULL;
    state->str_freelist_count = 0;
    state->strs_freelist_head = NULL;
    state->strs_freelist_count = 0;
}

static PyMethodDef stringzilla_methods[] = {
    // Basic `str`, `bytes`, and `bytearray`-like functionality
    {"contains", (PyCFunction)Str_like_contains, SZ_METHOD_FLAGS, doc_contains},
    {"count", (PyCFunction)Str_like_count, SZ_METHOD_FLAGS, doc_count},
    {"splitlines", (PyCFunction)Str_like_splitlines, SZ_METHOD_FLAGS, doc_splitlines},
    {"startswith", (PyCFunction)Str_like_startswith, SZ_METHOD_FLAGS, doc_startswith},
    {"endswith", (PyCFunction)Str_like_endswith, SZ_METHOD_FLAGS, doc_endswith},
    {"decode", (PyCFunction)Str_like_decode, SZ_METHOD_FLAGS, doc_decode},
    {"equal", (PyCFunction)Str_like_equal, SZ_METHOD_FLAGS, doc_like_equal},
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

    // Global unary extensions
    {"hash", (PyCFunction)Str_like_hash, SZ_METHOD_FLAGS, doc_like_hash},
    {"hash_multiseed", (PyCFunction)Str_like_hash_multiseed, SZ_METHOD_FLAGS, doc_hash_multiseed},
    {"bytesum", (PyCFunction)Str_like_bytesum, SZ_METHOD_FLAGS, doc_like_bytesum},
    {"sha256", (PyCFunction)Str_like_sha256, SZ_METHOD_FLAGS, doc_like_sha256},
    {"hmac_sha256", (PyCFunction)hmac_sha256, SZ_METHOD_FLAGS, doc_hmac_sha256},
    {"fill_random", (PyCFunction)Str_like_fill_random, SZ_METHOD_FLAGS, doc_fill_random},

    // Module-level functionality
    {"random", (PyCFunction)module_random, SZ_METHOD_FLAGS, doc_random},
    {"reset_capabilities", (PyCFunction)module_reset_capabilities, METH_VARARGS, doc_reset_capabilities},

    {NULL, NULL, 0, NULL}};

PyModuleDef stringzilla_module = {
    PyModuleDef_HEAD_INIT,
    "stringzilla",
    "Search, hash, sort, fingerprint, and fuzzy-match strings faster via SWAR, SIMD, and GPGPU",
    sizeof(stringzilla_state_t), // Per-interpreter free-list state; also enables `PyState_FindModule`.
    stringzilla_methods,
    NULL,
    NULL,
    NULL,
    stringzilla_cleanup,
};

/**
 *  @brief Every type the module readies, and the name it is exported under.
 *
 *  A `NULL` name means the type is readied but not exported: the UTF-8 iterators are returned by methods
 *  and never constructed by name, so they need `tp_dict` filled but no module attribute.
 */
static struct {
    char const *name;
    PyTypeObject *type;
} const stringzilla_types[] = {
    {"Str", &StrType},
    {"File", &FileType},
    {"Strs", &StrsType},
    {"FindSplits", &FindSplitsType},
    {"Utf8SplitNewlines", &Utf8SplitNewlinesType},
    {"Utf8Newlines", &Utf8NewlinesType},
    {"Utf8SplitWhitespaces", &Utf8SplitWhitespacesType},
    {"Utf8Whitespaces", &Utf8WhitespacesType},
    {"Utf8SplitDelimiters", &Utf8SplitDelimitersType},
    {"Utf8Delimiters", &Utf8DelimitersType},
    {"Utf8Wordbreaks", &Utf8WordbreaksType},
    {NULL, &Utf8CodepointsType},
    {NULL, &Utf8GraphemesType},
    {NULL, &Utf8SentencesType},
    {NULL, &Utf8LinebreaksType},
    {NULL, &Utf8UncasedMatchesType},
    {"Hasher", &HasherType},
    {"Sha256", &Sha256Type},
    {"Sha256s", &Sha256sType},
    {"Aes256CtrKey", &Aes256CtrKeyType},
    {"Aes256GcmKey", &Aes256GcmKeyType},
    {"Aes256GcmEncryptor", &Aes256GcmEncryptorType},
    {"Aes256GcmDecryptor", &Aes256GcmDecryptorType},
};

PyMODINIT_FUNC PyInit_stringzilla(void) {
    PyObject *m;
    sz_size_t const types_count = sizeof(stringzilla_types) / sizeof(stringzilla_types[0]);

    for (sz_size_t type_index = 0; type_index != types_count; ++type_index)
        if (PyType_Ready(stringzilla_types[type_index].type) < 0) return NULL;

    m = PyModule_Create(&stringzilla_module);
    if (m == NULL) return NULL;

#ifdef Py_GIL_DISABLED
    // Declare that this module is safe for free-threaded Python
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    // Add version metadata
    {
        char version_str[50];
        sprintf(version_str, "%d.%d.%d", sz_version_major(), sz_version_minor(), sz_version_patch());
        PyModule_AddStringConstant(m, "__version__", version_str);
    }

    // Publish the digest width on both hasher types, so callers can size an output matrix without
    // hardcoding it. A class attribute rather than a property, as it describes the algorithm.
    {
        PyObject *digest_length = PyLong_FromSize_t(SZ_SHA256_DIGEST_LENGTH);
        if (!digest_length) goto failed;
        int const published = PyDict_SetItemString(Sha256Type.tp_dict, "digest_length", digest_length) |
                              PyDict_SetItemString(Sha256sType.tp_dict, "digest_length", digest_length);
        Py_DECREF(digest_length);
        if (published < 0) goto failed;
    }

    // Define SIMD capabilities as a tuple
    {
        sz_capability_t caps = sz_capabilities();

        PyObject *caps_tuple = capabilities_to_tuple(caps);
        if (!caps_tuple) goto failed;

        if (PyModule_AddObject(m, "__capabilities__", caps_tuple) < 0) {
            Py_DECREF(caps_tuple);
            goto failed;
        }

        // Also keep the old comma-separated string version for backward compatibility
        sz_cptr_t caps_str = sz_capabilities_to_string(caps);
        PyModule_AddStringConstant(m, "__capabilities_str__", caps_str);
    }

    for (sz_size_t type_index = 0; type_index != types_count; ++type_index) {
        char const *const name = stringzilla_types[type_index].name;
        if (!name) continue;
        PyTypeObject *const type = stringzilla_types[type_index].type;
        Py_INCREF(type);
        // Only a successful `PyModule_AddObject` steals the reference, so undo just this one and let the
        // dying module release the types it already owns.
        if (PyModule_AddObject(m, name, (PyObject *)type) < 0) {
            Py_DECREF(type);
            goto failed;
        }
    }

    // A refused tag needs a class of its own, created here because a static exception object cannot
    // be built before the interpreter exists.
    AuthenticationErrorType = PyErr_NewExceptionWithDoc("stringzilla.AuthenticationError", doc_AuthenticationError,
                                                        PyExc_ValueError, NULL);
    if (!AuthenticationErrorType) goto failed;
    Py_INCREF(AuthenticationErrorType);
    if (PyModule_AddObject(m, "AuthenticationError", AuthenticationErrorType) < 0) {
        Py_DECREF(AuthenticationErrorType); // The reference the module refused to take
        Py_DECREF(AuthenticationErrorType); // The reference this translation unit holds
        AuthenticationErrorType = NULL;
        goto failed;
    }

    // Export C API functions as a single capsule structure for StringZillas
    static PyAPI sz_py_api = {
        .sz_py_export_string_like = sz_py_export_string_like,
        .sz_py_export_strings_as_sequence = sz_py_export_strings_as_sequence,
        .sz_py_export_strings_as_u32tape = sz_py_export_strings_as_u32tape,
        .sz_py_export_strings_as_u64tape = sz_py_export_strings_as_u64tape,
        .sz_py_replace_strings_allocator = sz_py_replace_strings_allocator,
    };
    if (PyModule_AddObject(m, "_sz_py_api", PyCapsule_New(&sz_py_api, "_sz_py_api", NULL)) < 0) goto failed;

    return m;

failed:
    Py_DECREF(m);
    return NULL;
}
