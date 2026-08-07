/**
 *  @brief Byte-sums, the 64-bit hasher, SHA-256 states, and HMAC-SHA256.
 *  @file python/stringzilla/hash.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

/**
 *  @brief Points @p sequence at the strings in @p texts, borrowing them in place.
 *
 *  A `Strs` is read through its own layout-aware accessors, so a tape-backed corpus never leaves its
 *  buffer. Any other sequence is snapshotted into a tuple and unpacked into @p scratch, which must hold
 *  @p scratch_capacity views. The snapshot is what pins the chunks: holding the caller's list would keep
 *  the container alive while leaving each chunk free to be dropped from under the exported pointers.
 *
 *  @return A new reference that pins the strings and must outlive the kernel call, or `NULL` on failure.
 */
static PyObject *Sha256_bind_texts_(PyObject *texts, sz_sequence_t *sequence, sz_string_view_t *scratch,
                                    sz_size_t scratch_capacity) {

    if (PyObject_TypeCheck(texts, &StrsType)) {
        sz_size_t const count = (sz_size_t)Strs_len((Strs *)texts);
        if (count > scratch_capacity) {
            PyErr_Format(PyExc_ValueError, "Expected exactly one chunk per lane, got %zu for %zu lanes", count,
                         scratch_capacity);
            return NULL;
        }
        sequence->handle = texts;
        sequence->count = count;
        sequence->get_start = Strs_get_start_;
        sequence->get_length = Strs_get_length_;
        Py_INCREF(texts);
        return texts;
    }

    PyObject *snapshot = PySequence_Tuple(texts);
    if (!snapshot) {
        wrap_current_exception("Argument must be a sequence of string-like chunks");
        return NULL;
    }

    sz_size_t const count = (sz_size_t)PyTuple_GET_SIZE(snapshot);
    if (count > scratch_capacity) {
        Py_DECREF(snapshot);
        PyErr_Format(PyExc_ValueError, "Expected exactly one chunk per lane, got %zu for %zu lanes", count,
                     scratch_capacity);
        return NULL;
    }
    for (sz_size_t index = 0; index != count; ++index)
        if (!sz_py_export_string_like(PyTuple_GET_ITEM(snapshot, (Py_ssize_t)index), &scratch[index].start,
                                      &scratch[index].length)) {
            Py_DECREF(snapshot);
            wrap_current_exception("Every chunk must be string-like");
            return NULL;
        }

    sz_sequence_from_string_views(scratch, count, sequence);
    return snapshot;
}

/**
 *  @brief Binds a writable buffer of at least @p digests_count digests, or reports why it can't.
 *  @return Whether @p out_view was filled and must later be released.
 */
static sz_bool_t Sha256_bind_digests_(PyObject *out, sz_size_t digests_count, Py_buffer *out_view) {
    // PyBUF_CONTIG = writable + C-contiguous; rejects strided and read-only targets up front.
    if (PyObject_GetBuffer(out, out_view, PyBUF_CONTIG) != 0) return sz_false_k;
    if (out_view->itemsize != 1) {
        PyErr_SetString(PyExc_TypeError, "out must be a contiguous buffer of bytes (e.g. numpy.uint8)");
        PyBuffer_Release(out_view);
        return sz_false_k;
    }
    if ((sz_size_t)out_view->len < digests_count * SZ_SHA256_DIGEST_LENGTH) {
        PyErr_Format(PyExc_ValueError, "out buffer holds %zd bytes, need %zu for %zu digests",
                     (Py_ssize_t)out_view->len, digests_count * SZ_SHA256_DIGEST_LENGTH, digests_count);
        PyBuffer_Release(out_view);
        return sz_false_k;
    }
    return sz_true_k;
}

/** @brief Packs @p digests_count digests into a list of `bytes`, in lane order. */
static PyObject *Sha256_digests_to_list_(sz_u8_t const *digests, sz_size_t digests_count) {
    PyObject *result = PyList_New((Py_ssize_t)digests_count);
    if (!result) return NULL;
    for (sz_size_t index = 0; index != digests_count; ++index) {
        PyObject *digest = PyBytes_FromStringAndSize((char const *)&digests[index * SZ_SHA256_DIGEST_LENGTH],
                                                     SZ_SHA256_DIGEST_LENGTH);
        if (!digest) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, (Py_ssize_t)index, digest);
    }
    return result;
}

Py_hash_t Str_hash(Str *self) { return (Py_hash_t)sz_hash(self->memory.start, self->memory.length, 0); }

char const doc_like_hash[] =                                                                             //
    "Compute the hash value of the string.\n"                                                            //
    "\n"                                                                                                 //
    "This function can be called as a method on a Str object or as a standalone function.\n"             //
    "Args:\n"                                                                                            //
    "  text (Str or str or bytes): The string to hash (positional-only when standalone).\n"              //
    "  seed (int, optional): The seed value for hashing. Defaults to 0. Can be positional or keyword.\n" //
    "Returns:\n"                                                                                         //
    "  int: The hash value as an unsigned 64-bit integer. This differs from Python's\n"                  //
    "       built-in `hash()` which returns a `Py_hash_t` and may be platform-dependent.\n"              //
    "Raises:\n"                                                                                          //
    "  TypeError: If the argument is not string-like or incorrect number of arguments is provided.\n"    //
    "Signature:\n"                                                                                       //
    "  >>> def hash(text, seed=0, /) -> int: ...";

PyObject *Str_like_hash(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                        PyObject *args_names_tuple) {
    // Fast path variables
    PyObject *text_obj = NULL;
    PyObject *seed_obj = NULL;
    sz_string_view_t text;
    sz_u64_t seed = 0;

    // Check if this is a method call on a Str instance
    int const is_member = self != NULL && PyObject_TypeCheck(self, &StrType);

    // Fast argument validation
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const total_args = positional_args_count + args_names_count;
    Py_ssize_t const expected_min = is_member ? 0 : 1;
    Py_ssize_t const expected_max = expected_min + 1;

    if (total_args < expected_min || total_args > expected_max) {
        PyErr_SetString(PyExc_TypeError, is_member ? "hash() takes 0 or 1 positional arguments"
                                                   : "hash() takes 1 or 2 positional arguments");
        return NULL;
    }

    if (positional_args_count > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Too many positional arguments");
        return NULL;
    }

    // Fast positional argument extraction
    if (is_member) {
        text_obj = self;
        if (positional_args_count >= 1) seed_obj = args[0];
    }
    else {
        if (positional_args_count >= 1) text_obj = args[0];
        if (positional_args_count >= 2) seed_obj = args[1];
    }

    // Fast keyword argument parsing
    if (args_names_count > 0) {
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *const key = PyTuple_GetItem(args_names_tuple, i);
            PyObject *const value = args[positional_args_count + i];

            if (PyUnicode_CompareWithASCIIString(key, "seed") == 0) {
                if (seed_obj) {
                    PyErr_SetString(PyExc_TypeError, "seed specified twice");
                    return NULL;
                }
                seed_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", key);
                return NULL;
            }
        }
    }

    // Validate and convert text
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    // Parse seed
    if (seed_obj) {
        if (!PyLong_Check(seed_obj)) {
            PyErr_SetString(PyExc_TypeError, "seed must be an integer");
            return NULL;
        }
        seed = PyLong_AsUnsignedLongLong(seed_obj);
        if (PyErr_Occurred()) return NULL;
    }

    sz_u64_t result = sz_hash(text.start, text.length, seed);
    return PyLong_FromUnsignedLongLong((unsigned long long)result);
}

char const doc_hash_multiseed[] =                                                                            //
    "Hash one string under many seeds at once.\n"                                                            //
    "\n"                                                                                                     //
    "Equivalent to `tuple(hash(text, s) for s in seeds)`, but normalizes the input into AES\n"               //
    "blocks once and replays the cheap per-seed rounds - markedly faster for short strings under\n"          //
    "many seeds (feature hashing, Count-Min sketches, Bloom/cuckoo filters, MinHash/LSH).\n"                 //
    "\n"                                                                                                     //
    "Args:\n"                                                                                                //
    "  text (Str or str or bytes): The string to hash (positional-only when standalone).\n"                  //
    "  seeds (buffer of uint64): Contiguous 64-bit seeds, e.g. numpy.uint64 array or array('Q', ...).\n"     //
    "       Plain int lists are not accepted - wrap them in array('Q', seeds) or numpy.\n"                   //
    "  out (writable uint64 buffer, optional): A contiguous buffer of at least len(seeds) elements.\n"       //
    "       When given it is filled in place and None is returned; otherwise a tuple of ints is returned.\n" //
    "Returns:\n"                                                                                             //
    "  tuple[int, ...] | None: Tuple of 64-bit hashes, or None when writing into `out`.\n"                   //
    "\n"                                                                                                     //
    "Example:\n"                                                                                             //
    "  >>> def hash_multiseed(text, seeds, /, out=None) -> tuple | None: ...";

PyObject *Str_like_hash_multiseed(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *args_names_tuple) {
    int const is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    PyObject *text_obj = is_member ? self : NULL;
    PyObject *seeds_obj = NULL;
    PyObject *out_obj = NULL;

    // Positional layout: [text if standalone], seeds, [out]
    Py_ssize_t const seeds_slot = is_member ? 0 : 1;
    if (!is_member && positional_args_count >= 1) text_obj = args[0];
    if (positional_args_count > seeds_slot) seeds_obj = args[seeds_slot];
    if (positional_args_count > seeds_slot + 1) out_obj = args[seeds_slot + 1];

    // Keyword arguments: `seeds`, `out`
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t i = 0; i < args_names_count; ++i) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, i);
        PyObject *const value = args[positional_args_count + i];
        if (PyUnicode_CompareWithASCIIString(key, "seeds") == 0 && !seeds_obj) seeds_obj = value;
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) out_obj = value;
        else {
            PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", key);
            return NULL;
        }
    }

    if (!text_obj || !seeds_obj) {
        PyErr_SetString(PyExc_TypeError, "hash_multiseed() requires a text and a sequence of seeds");
        return NULL;
    }

    sz_string_view_t text;
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    // Acquire the seeds as a contiguous, read-only 64-bit buffer (e.g. NumPy uint64, array('Q'),
    // memoryview) - zero-copy, no allocation, no per-element boxing. Plain int lists are rejected.
    Py_buffer seeds_view;
    if (PyObject_GetBuffer(seeds_obj, &seeds_view, PyBUF_CONTIG_RO | PyBUF_FORMAT) != 0) return NULL;
    if (seeds_view.itemsize != (Py_ssize_t)sizeof(sz_u64_t)) {
        PyErr_SetString(PyExc_TypeError,
                        "seeds must be a contiguous buffer of 64-bit integers (e.g. numpy.uint64 or array('Q'))");
        PyBuffer_Release(&seeds_view);
        return NULL;
    }
    sz_size_t const count = (sz_size_t)(seeds_view.len / (Py_ssize_t)sizeof(sz_u64_t));
    sz_u64_t const *seeds = (sz_u64_t const *)seeds_view.buf;

    // Write the hashes into the caller buffer (zero allocation), or build a tuple via a stack scratch.
    if (out_obj && out_obj != Py_None) {
        Py_buffer out_view;
        // PyBUF_CONTIG = writable + C-contiguous; rejects strided/non-contiguous targets up front.
        if (PyObject_GetBuffer(out_obj, &out_view, PyBUF_CONTIG) != 0) {
            PyBuffer_Release(&seeds_view);
            return NULL;
        }
        if ((size_t)out_view.len < (size_t)count * sizeof(sz_u64_t)) {
            PyErr_Format(PyExc_ValueError, "out buffer holds %zd bytes, need %zu for %zu seeds",
                         (Py_ssize_t)out_view.len, (size_t)count * sizeof(sz_u64_t), (size_t)count);
            PyBuffer_Release(&out_view);
            PyBuffer_Release(&seeds_view);
            return NULL;
        }
        if (count) sz_hash_multiseed(text.start, text.length, seeds, count, (sz_u64_t *)out_view.buf);
        PyBuffer_Release(&out_view);
        PyBuffer_Release(&seeds_view);
        Py_RETURN_NONE;
    }

    PyObject *result = PyTuple_New((Py_ssize_t)count);
    if (!result) {
        PyBuffer_Release(&seeds_view);
        return NULL;
    }
    sz_u64_t scratch[256];
    for (sz_size_t offset = 0; offset < count; offset += 256) {
        sz_size_t const chunk = count - offset < 256 ? count - offset : 256;
        sz_hash_multiseed(text.start, text.length, seeds + offset, chunk, scratch);
        for (sz_size_t i = 0; i < chunk; ++i) {
            PyObject *item = PyLong_FromUnsignedLongLong((unsigned long long)scratch[i]);
            if (!item) {
                Py_DECREF(result);
                PyBuffer_Release(&seeds_view);
                return NULL;
            }
            PyTuple_SET_ITEM(result, (Py_ssize_t)(offset + i), item);
        }
    }
    PyBuffer_Release(&seeds_view);
    return result;
}

char const doc_fill_random[] =                                                                                 //
    "Fill a string-like buffer in place with pseudo-random bytes.\n"                                           //
    "\n"                                                                                                       //
    "Args:\n"                                                                                                  //
    "  buffer (Str or bytes-like): Writable, contiguous byte buffer (e.g., memoryview/bytearray).\n"           //
    "  nonce (int, optional): Seed/nonce ensuring reproducible output for the same inputs (default 0).\n"      //
    "  alphabet (str or bytes, optional): If provided, remaps random bytes to characters from the alphabet.\n" //
    "  start (int, optional): Starting index (default 0).\n"                                                   //
    "  end (int, optional): Ending index (default len(buffer)).\n"                                             //
    "Returns:\n"                                                                                               //
    "  None: Mutates the buffer slice in place.\n"                                                             //
    "\n"                                                                                                       //
    "Example:\n"                                                                                               //
    "  >>> buf = bytearray(16)\n"                                                                              //
    "  >>> sz.fill_random(buf)\n"                                                                              //
    "  >>> len(buf)\n"                                                                                         //
    "  16";

PyObject *Str_like_fill_random(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                               PyObject *args_names_tuple) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count < !is_member || positional_args_count > !is_member + 3) {
        PyErr_SetString(PyExc_TypeError, "fill_random() expects 1 to 4 positional arguments");
        return NULL;
    }

    PyObject *buffer_obj = is_member ? self : args[0];
    PyObject *nonce_obj = positional_args_count > !is_member ? args[!is_member] : NULL;
    PyObject *start_obj = positional_args_count > !is_member + 1 ? args[!is_member + 1] : NULL;
    PyObject *end_obj = positional_args_count > !is_member + 2 ? args[!is_member + 2] : NULL;
    PyObject *alphabet_obj = NULL;

    // Optional keyword arguments
    if (args_names_tuple) {
        Py_ssize_t kw_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < kw_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "nonce") == 0 && !nonce_obj) nonce_obj = value;
            else if (PyUnicode_CompareWithASCIIString(key, "alphabet") == 0 && !alphabet_obj) alphabet_obj = value;
            else if (PyUnicode_CompareWithASCIIString(key, "start") == 0 && !start_obj) start_obj = value;
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0 && !end_obj) end_obj = value;
            else {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", key);
                return NULL;
            }
        }
    }

    // Parse start/end
    Py_ssize_t start = 0, end = PY_SSIZE_T_MAX;
    if (start_obj && ((start = PyLong_AsSsize_t(start_obj)) == -1 && PyErr_Occurred())) {
        PyErr_SetString(PyExc_TypeError, "start must be an integer");
        return NULL;
    }
    if (end_obj && ((end = PyLong_AsSsize_t(end_obj)) == -1 && PyErr_Occurred())) {
        PyErr_SetString(PyExc_TypeError, "end must be an integer");
        return NULL;
    }

    // Parse nonce
    sz_u64_t nonce = 0;
    if (nonce_obj) {
        if (!PyLong_Check(nonce_obj)) {
            PyErr_SetString(PyExc_TypeError, "nonce must be an integer");
            return NULL;
        }
        nonce = PyLong_AsUnsignedLongLong(nonce_obj);
        if (PyErr_Occurred()) return NULL;
    }

    // Parse alphabet
    sz_string_view_t alphabet;
    if (alphabet_obj) {
        if (!sz_py_export_string_like(alphabet_obj, &alphabet.start, &alphabet.length)) {
            wrap_current_exception("alphabet must be string-like");
            return NULL;
        }
        if (alphabet.length == 0) {
            PyErr_SetString(PyExc_ValueError, "alphabet must not be empty");
            return NULL;
        }
    }

    // Export buffer and clamp range
    sz_string_view_t buf;
    if (!sz_py_export_string_like(buffer_obj, &buf.start, &buf.length)) {
        wrap_current_exception("First argument must be string-like");
        return NULL;
    }

    if (sz_py_is_mutable(buffer_obj) == sz_false_k) return NULL;

    if (start < 0 || (end != PY_SSIZE_T_MAX && end < 0)) {
        PyErr_SetString(PyExc_ValueError, "start/end must be non-negative");
        return NULL;
    }

    if ((sz_size_t)start > buf.length) {
        Py_RETURN_NONE; // nothing to do
    }

    buf.start += start;
    buf.length -= start;
    if (end != PY_SSIZE_T_MAX) {
        sz_size_t window_length = end >= start ? (sz_size_t)(end - start) : 0; // Inverted window fills nothing
        if (window_length < buf.length) buf.length = window_length;
    }

    sz_fill_random((sz_ptr_t)buf.start, buf.length, nonce);
    if (alphabet_obj) {
        sz_align_(64) char look_up_table[256];
        for (int i = 0; i < 256; ++i) look_up_table[i] = alphabet.start[i % alphabet.length];
        sz_lookup((sz_ptr_t)buf.start, buf.length, (sz_cptr_t)buf.start, look_up_table);
    }
    Py_RETURN_NONE;
}

char const doc_random[] =                                                            //
    "random(length, *, nonce=0, alphabet=None) -> bytes\n\n"                         //
    "Generate a new random byte string, optionally remapped to a given alphabet.\n"  //
    "If alphabet is provided, each byte is mapped to alphabet[b % len(alphabet)].\n" //
    "\n"                                                                             //
    "Example:\n"                                                                     //
    "  >>> len(sz.random(16))\n"                                                     //
    "  16";

PyObject *module_random(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                        PyObject *args_names_tuple) {
    sz_unused_(self);
    if (positional_args_count < 1 || positional_args_count > 2) {
        PyErr_SetString(PyExc_TypeError, "random() expects 1 or 2 positional arguments");
        return NULL;
    }
    PyObject *length_obj = args[0];
    PyObject *nonce_obj = positional_args_count > 1 ? args[1] : NULL;
    PyObject *alphabet_obj = NULL;

    if (args_names_tuple) {
        Py_ssize_t kw_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < kw_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "nonce") == 0 && !nonce_obj) nonce_obj = value;
            else if (PyUnicode_CompareWithASCIIString(key, "alphabet") == 0 && !alphabet_obj) alphabet_obj = value;
            else {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", key);
                return NULL;
            }
        }
    }

    if (!PyLong_Check(length_obj)) {
        PyErr_SetString(PyExc_TypeError, "length must be an integer");
        return NULL;
    }
    Py_ssize_t signed_length = PyLong_AsSsize_t(length_obj);
    if (signed_length == -1 && PyErr_Occurred()) return NULL;
    if (signed_length < 0) {
        PyErr_SetString(PyExc_ValueError, "length must be non-negative");
        return NULL;
    }
    sz_size_t length = (sz_size_t)signed_length;

    sz_u64_t nonce = 0;
    if (nonce_obj) {
        if (!PyLong_Check(nonce_obj)) {
            PyErr_SetString(PyExc_TypeError, "nonce must be an integer");
            return NULL;
        }
        nonce = PyLong_AsUnsignedLongLong(nonce_obj);
        if (PyErr_Occurred()) return NULL;
    }

    PyObject *bytes_obj = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)length);
    if (!bytes_obj) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate random bytes");
        return NULL;
    }
    if (length > 0) {
        sz_ptr_t buffer = (sz_ptr_t)PyBytes_AS_STRING(bytes_obj);
        sz_fill_random(buffer, length, nonce);
    }

    if (!alphabet_obj || length == 0) return bytes_obj;

    sz_string_view_t alphabet;
    if (!sz_py_export_string_like(alphabet_obj, &alphabet.start, &alphabet.length)) {
        Py_DECREF(bytes_obj);
        wrap_current_exception("alphabet must be string-like");
        return NULL;
    }
    if (alphabet.length == 0) {
        Py_DECREF(bytes_obj);
        PyErr_SetString(PyExc_ValueError, "alphabet must not be empty");
        return NULL;
    }

    sz_align_(64) char look_up_table[256];
    for (int i = 0; i < 256; ++i) look_up_table[i] = alphabet.start[i % alphabet.length];
    sz_ptr_t buf_ptr = (sz_ptr_t)PyBytes_AS_STRING(bytes_obj);
    sz_lookup(buf_ptr, length, buf_ptr, look_up_table);
    return bytes_obj;
}

char const doc_like_bytesum[] =                                                                       //
    "Compute the checksum of individual byte values in a string.\n"                                   //
    "\n"                                                                                              //
    "This function can be called as a method on a Str object or as a standalone function.\n"          //
    "Args:\n"                                                                                         //
    "  text (Str or str or bytes): The string to hash.\n"                                             //
    "Returns:\n"                                                                                      //
    "  int: The checksum of individual byte values in a string.\n"                                    //
    "Raises:\n"                                                                                       //
    "  TypeError: If the argument is not string-like or incorrect number of arguments is provided.\n" //
    "\n"                                                                                              //
    "Example:\n"                                                                                      //
    "  >>> sz.bytesum('abc')\n"                                                                       //
    "  294";

PyObject *Str_like_bytesum(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                           PyObject *args_names_tuple) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count < !is_member || positional_args_count > !is_member + 1 || args_names_tuple) {
        PyErr_SetString(PyExc_TypeError, "bytesum() expects exactly one positional argument");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    sz_string_view_t text;

    // Validate and convert `text`
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    sz_u64_t result = sz_bytesum(text.start, text.length);
    return PyLong_FromUnsignedLongLong((unsigned long long)result);
}

char const doc_like_sha256[] =                                                                        //
    "Compute SHA256 cryptographic hash of the input data.\n"                                          //
    "\n"                                                                                              //
    "This function can be called as a method on a Str object or as a standalone function.\n"          //
    "Args:\n"                                                                                         //
    "  text (Str or str or bytes): The input data to hash.\n"                                         //
    "Returns:\n"                                                                                      //
    "  bytes: The 32-byte (256-bit) SHA256 digest.\n"                                                 //
    "Raises:\n"                                                                                       //
    "  TypeError: If the argument is not string-like or incorrect number of arguments is provided.\n" //
    "\n"                                                                                              //
    "Example:\n"                                                                                      //
    "  >>> sz.Str('abc').sha256().hex()[:8]\n"                                                        //
    "  'ba7816bf'";

PyObject *Str_like_sha256(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                          PyObject *args_names_tuple) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count < !is_member || positional_args_count > !is_member + 1 || args_names_tuple) {
        PyErr_SetString(PyExc_TypeError, "sha256() expects exactly one positional argument");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    sz_string_view_t text;

    // Validate and convert `text`
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    // Initialize SHA256 state
    sz_sha256_state_t state;
    sz_sha256_state_init(&state);

    // Update with data
    sz_sha256_state_update(&state, text.start, text.length);

    // Compute final digest
    sz_u8_t digest[32];
    sz_sha256_state_digest(&state, digest);

    return PyBytes_FromStringAndSize((char const *)digest, 32);
}

/**
 *  @brief Primes @p inner and @p outer with the HMAC key pads, per FIPS 198-1.
 *
 *  The message enters only the inner hash, as the suffix of the ipad block, so authenticating many
 *  messages under one key is just as lane-parallel as digesting them: prime once, broadcast into every
 *  lane, and pay for the key again only in the outer wrap.
 */
static void Hmac_prime_(sz_sha256_state_t *inner, sz_sha256_state_t *outer, sz_cptr_t key, sz_size_t key_length) {

    // Keys longer than one block are replaced by their digest; shorter ones are zero-padded.
    sz_u8_t key_pad[SZ_SHA256_BLOCK_LENGTH];
    sz_fill((sz_ptr_t)key_pad, sizeof(key_pad), 0);
    if (key_length > SZ_SHA256_BLOCK_LENGTH) {
        sz_sha256_state_t key_state;
        sz_sha256_state_init(&key_state);
        sz_sha256_state_update(&key_state, key, key_length);
        sz_sha256_state_digest(&key_state, key_pad);
    }
    else { sz_copy((sz_ptr_t)key_pad, key, key_length); }

    sz_u8_t block[SZ_SHA256_BLOCK_LENGTH];
    sz_sha256_state_init(inner);
    for (sz_size_t byte_index = 0; byte_index != SZ_SHA256_BLOCK_LENGTH; ++byte_index)
        block[byte_index] = key_pad[byte_index] ^ 0x36;
    sz_sha256_state_update(inner, (sz_cptr_t)block, SZ_SHA256_BLOCK_LENGTH);

    sz_sha256_state_init(outer);
    for (sz_size_t byte_index = 0; byte_index != SZ_SHA256_BLOCK_LENGTH; ++byte_index)
        block[byte_index] = key_pad[byte_index] ^ 0x5c;
    sz_sha256_state_update(outer, (sz_cptr_t)block, SZ_SHA256_BLOCK_LENGTH);

    // The pads are derived from the secret, so don't leave them on the stack for the next frame.
    sz_fill((sz_ptr_t)key_pad, sizeof(key_pad), 0), sz_fill((sz_ptr_t)block, sizeof(block), 0);
}

/**
 *  @brief Wraps @p inner's digest with the primed @p outer state, completing one tag.
 *  @note Consumes neither state, so a caller can take an interim tag and keep streaming.
 */
static void Hmac_digest_one_(sz_sha256_state_t const *inner, sz_sha256_state_t const *outer, sz_u8_t *digest) {
    sz_u8_t inner_digest[SZ_SHA256_DIGEST_LENGTH];
    sz_sha256_state_digest(inner, inner_digest);
    sz_sha256_state_t wrapping = *outer;
    sz_sha256_state_update(&wrapping, (sz_cptr_t)inner_digest, SZ_SHA256_DIGEST_LENGTH);
    sz_sha256_state_digest(&wrapping, digest);
}

/**
 *  @brief Authenticates @p texts under one key, writing one tag per message into @p digests.
 *
 *  Runs the message pass and the outer wrap through the same lane-parallel kernels, reusing @p states
 *  for both so the wrap costs no extra allocation.
 *
 *  @param states Scratch of at least `texts->count` states.
 *  @param views Scratch of at least `texts->count` views, reused for the inner digests.
 *  @note Runs with the GIL released, so it must touch no Python object.
 */
static void Hmac_digest_many_(sz_sequence_t const *texts, sz_sha256_state_t const *inner,
                              sz_sha256_state_t const *outer, sz_sha256_state_t *states, sz_string_view_t *views,
                              sz_u8_t *digests) {

    sz_size_t const count = texts->count;
    for (sz_size_t index = 0; index != count; ++index) states[index] = *inner;
    sz_sha256_multistate_update(states, texts);
    sz_sha256_multistate_digest(states, count, digests);

    for (sz_size_t index = 0; index != count; ++index)
        states[index] = *outer, views[index].start = (sz_cptr_t)&digests[index * SZ_SHA256_DIGEST_LENGTH],
        views[index].length = SZ_SHA256_DIGEST_LENGTH;

    sz_sequence_t inner_digests;
    sz_sequence_from_string_views(views, count, &inner_digests);
    sz_sha256_multistate_update(states, &inner_digests);

    // Safe to overwrite in place: every inner digest has already been absorbed into its lane.
    sz_sha256_multistate_digest(states, count, digests);
}

char const doc_hmac_sha256[] =                                                                       //
    "hmac_sha256(key, message, out=None) -> bytes | list[bytes] | buffer\n"                          //
    "\n"                                                                                             //
    "Compute the HMAC-SHA256 authentication code of one message, or of many under one key.\n"        //
    "\n"                                                                                             //
    "Authenticating one message is a serial dependency chain no instruction set can speed up, but\n" //
    "independent messages compress in parallel lanes: sixteen at a time on AVX-512, eight on\n"      //
    "AVX2. Pass a Strs of messages - a batch of tokens or requests sharing a secret - and both\n"    //
    "the message pass and the outer wrap run lane-parallel.\n"                                       //
    "\n"                                                                                             //
    "Args:\n"                                                                                        //
    "  key (str | bytes): The secret key.\n"                                                         //
    "  message (str | bytes | Strs | sequence): One message, or many to authenticate together.\n"    //
    "  out (buffer, optional): Writable, C-contiguous buffer of bytes holding at least 32 per\n"     //
    "       message - a `(len(message), 32)` numpy.uint8 matrix receives one tag per row with\n"     //
    "       zero allocation. Defaults to None.\n"                                                    //
    "Returns:\n"                                                                                     //
    "  bytes: The 32-byte tag, for a single string-like message.\n"                                  //
    "  list[bytes]: One tag per message, in order, for a collection.\n"                              //
    "  buffer: `out` itself, when an output buffer is given.\n"                                      //
    "Example:\n"                                                                                     //
    "  >>> len(sz.hmac_sha256(b'key', b'message'))\n"                                                //
    "  32\n"                                                                                         //
    "  >>> sz.hmac_sha256(b'key', sz.Strs(['a', 'b'])) == [\n"                                       //
    "  ...     sz.hmac_sha256(b'key', b'a'), sz.hmac_sha256(b'key', b'b')]\n"                        //
    "  True";

PyObject *hmac_sha256(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                      PyObject *args_names_tuple) {
    sz_unused_(self);
    PyObject *key_obj = NULL, *message_obj = NULL, *out_obj = NULL;
    if (positional_args_count > 3) {
        PyErr_SetString(PyExc_TypeError, "hmac_sha256() takes at most three positional arguments");
        return NULL;
    }
    if (positional_args_count >= 1) key_obj = args[0];
    if (positional_args_count >= 2) message_obj = args[1];
    if (positional_args_count == 3) out_obj = args[2];

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t i = 0; i < args_names_count; ++i) {
        PyObject *const name = PyTuple_GET_ITEM(args_names_tuple, i);
        PyObject *const value = args[positional_args_count + i];
        // Letting a duplicate silently win would pick a key the caller did not intend, so reject it.
        if (PyUnicode_CompareWithASCIIString(name, "key") == 0) {
            if (key_obj) {
                PyErr_SetString(PyExc_TypeError, "key specified twice");
                return NULL;
            }
            key_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "message") == 0) {
            if (message_obj) {
                PyErr_SetString(PyExc_TypeError, "message specified twice");
                return NULL;
            }
            message_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "out") == 0) {
            if (out_obj) {
                PyErr_SetString(PyExc_TypeError, "out specified twice");
                return NULL;
            }
            out_obj = value;
        }
        else {
            PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    // Validate all required arguments are provided
    if (!key_obj || !message_obj) {
        PyErr_SetString(PyExc_TypeError, "hmac_sha256() missing required arguments");
        return NULL;
    }

    sz_string_view_t key;
    if (!sz_py_export_string_like(key_obj, &key.start, &key.length)) {
        wrap_current_exception("Key must be string-like");
        return NULL;
    }
    sz_bool_t const have_out = (out_obj && out_obj != Py_None) ? sz_true_k : sz_false_k;

    // A string-like message is one message; anything else is a collection of them.
    sz_string_view_t message;
    if (sz_py_export_string_like(message_obj, &message.start, &message.length)) {
        Py_buffer out_view;
        if (have_out && !Sha256_bind_digests_(out_obj, 1, &out_view)) return NULL;

        sz_sha256_state_t inner_state, outer_state;
        sz_u8_t digest[SZ_SHA256_DIGEST_LENGTH];
        Hmac_prime_(&inner_state, &outer_state, key.start, key.length);
        sz_sha256_state_update(&inner_state, message.start, message.length);
        Hmac_digest_one_(&inner_state, &outer_state, have_out ? (sz_u8_t *)out_view.buf : digest);

        if (!have_out) return PyBytes_FromStringAndSize((char const *)digest, SZ_SHA256_DIGEST_LENGTH);
        PyBuffer_Release(&out_view);
        Py_INCREF(out_obj);
        return out_obj;
    }
    PyErr_Clear();

    Py_ssize_t const messages_count = PyObject_Size(message_obj);
    if (messages_count < 0) {
        PyErr_SetString(PyExc_TypeError, "Message must be string-like or a sequence of string-like messages");
        return NULL;
    }

    sz_string_view_t *views = NULL;
    sz_sha256_state_t *states = NULL;
    if (messages_count) {
        views = (sz_string_view_t *)malloc((size_t)messages_count * sizeof(sz_string_view_t));
        states = (sz_sha256_state_t *)malloc((size_t)messages_count * sizeof(sz_sha256_state_t));
        if (!views || !states) {
            free(views), free(states);
            return PyErr_NoMemory();
        }
    }

    sz_sequence_t texts;
    PyObject *pin = Sha256_bind_texts_(message_obj, &texts, views, (sz_size_t)messages_count);
    if (!pin) {
        free(views), free(states);
        return NULL;
    }

    Py_buffer out_view;
    if (have_out && !Sha256_bind_digests_(out_obj, texts.count, &out_view)) {
        Py_DECREF(pin);
        free(views), free(states);
        return NULL;
    }

    sz_u8_t *digests = NULL;
    if (have_out) { digests = (sz_u8_t *)out_view.buf; }
    else if (texts.count) {
        digests = (sz_u8_t *)malloc((size_t)texts.count * SZ_SHA256_DIGEST_LENGTH);
        if (!digests) {
            Py_DECREF(pin);
            free(views), free(states);
            return PyErr_NoMemory();
        }
    }

    if (texts.count) {
        // The `Strs` fast path leaves `views` unfilled, so materialize the spans the wrap will reuse.
        for (sz_size_t index = 0; index != texts.count; ++index)
            views[index].start = texts.get_start(texts.handle, index),
            views[index].length = texts.get_length(texts.handle, index);
        sz_sequence_from_string_views(views, texts.count, &texts);

        sz_sha256_state_t inner_state, outer_state;
        Hmac_prime_(&inner_state, &outer_state, key.start, key.length);
        Py_BEGIN_ALLOW_THREADS;
        Hmac_digest_many_(&texts, &inner_state, &outer_state, states, views, digests);
        Py_END_ALLOW_THREADS;
    }

    sz_size_t const digests_count = texts.count;
    Py_DECREF(pin);
    free(views), free(states);
    if (have_out) {
        PyBuffer_Release(&out_view);
        Py_INCREF(out_obj);
        return out_obj;
    }
    PyObject *result = Sha256_digests_to_list_(digests, digests_count);
    free(digests);
    return result;
}

typedef struct {
    PyObject ob_base;
    sz_hash_state_t state;
    sz_u64_t seed;
} Hasher;

static void Hasher_dealloc(Hasher *self) { Py_TYPE(self)->tp_free((PyObject *)self); }

static PyObject *Hasher_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    sz_unused_(args);
    sz_unused_(kwds);
    Hasher *self = (Hasher *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->seed = 0;
    sz_hash_state_init(&self->state, self->seed);
    return (PyObject *)self;
}

static int Hasher_init(Hasher *self, PyObject *args, PyObject *kwargs) {
    // Positional seed
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "Hasher() takes at most 1 positional argument");
        return -1;
    }
    PyObject *seed_obj = nargs == 1 ? PyTuple_GET_ITEM(args, 0) : NULL;
    // Keyword seed
    if (kwargs) {
        PyObject *kw_seed = PyDict_GetItemString(kwargs, "seed");
        if (kw_seed) {
            if (seed_obj) {
                PyErr_SetString(PyExc_TypeError, "seed specified twice");
                return -1;
            }
            seed_obj = kw_seed;
        }
        // Check for unexpected kwargs
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "seed") != 0) {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", key);
                return -1;
            }
        }
    }
    unsigned long long seed = 0ULL;
    if (seed_obj) {
        if (!PyLong_Check(seed_obj)) {
            PyErr_SetString(PyExc_TypeError, "seed must be an integer");
            return -1;
        }
        seed = PyLong_AsUnsignedLongLong(seed_obj);
        if (PyErr_Occurred()) return -1;
    }
    self->seed = (sz_u64_t)seed;
    sz_hash_state_init(&self->state, self->seed);
    return 0;
}

static PyObject *Hasher_update(PyObject *self_obj, PyObject *arg) {
    Hasher *self = (Hasher *)self_obj;
    sz_string_view_t text;
    if (!sz_py_export_string_like(arg, &text.start, &text.length)) {
        wrap_current_exception("Argument must be string-like");
        return NULL;
    }
    sz_hash_state_update(&self->state, text.start, text.length);
    Py_INCREF(self_obj);
    return self_obj;
}

static PyObject *Hasher_digest(PyObject *self_obj, PyObject *noargs) {
    sz_unused_(noargs);
    Hasher *self = (Hasher *)self_obj;
    sz_u64_t hash = sz_hash_state_digest(&self->state);
    return PyLong_FromUnsignedLongLong((unsigned long long)hash);
}

static PyObject *Hasher_hexdigest(PyObject *self_obj, PyObject *noargs) {
    sz_unused_(noargs);
    Hasher *self = (Hasher *)self_obj;
    sz_u64_t hash = sz_hash_state_digest(&self->state);
    char buf[17]; // lowercase, zero-padded 16 hex digits
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)hash);
    return PyUnicode_FromString(buf);
}

static PyObject *Hasher_reset(PyObject *self_obj, PyObject *noargs) {
    sz_unused_(noargs);
    Hasher *self = (Hasher *)self_obj;
    sz_hash_state_init(&self->state, self->seed);
    Py_INCREF(self_obj);
    return self_obj;
}

static char const doc_Hasher[] =                                                                  //
    "Hasher(seed=0)\n"                                                                            //
    "\n"                                                                                          //
    "Incremental, AES-accelerated hasher producing the same 64-bit value as `sz.hash`.\n"         //
    "Feed data in chunks with `update()`, read the digest, and reuse the object via `reset()`.\n" //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  seed (int, optional): 64-bit seed mixed into the initial state. Defaults to 0.\n"          //
    "Example:\n"                                                                                  //
    "  >>> h = sz.Hasher()\n"                                                                     //
    "  >>> _ = h.update(b'hello').update(b' world')\n"                                            //
    "  >>> h.digest() == sz.hash(b'hello world')\n"                                               //
    "  True";

static char const doc_Hasher_update[] =                                          //
    "update(data) -> Hasher\n"                                                   //
    "\n"                                                                         //
    "Absorb more bytes into the running hash and return self for chaining.\n"    //
    "\n"                                                                         //
    "Args:\n"                                                                    //
    "  data (str | bytes): Chunk to mix into the state.\n"                       //
    "Returns:\n"                                                                 //
    "  Hasher: The same object, enabling `h.update(a).update(b)`.\n"             //
    "Example:\n"                                                                 //
    "  >>> sz.Hasher().update(b'ab').update(b'c').digest() == sz.hash(b'abc')\n" //
    "  True";

static char const doc_Hasher_digest[] =                                                //
    "digest() -> int\n"                                                                //
    "\n"                                                                               //
    "Return the current hash as an unsigned 64-bit integer without consuming state.\n" //
    "\n"                                                                               //
    "Returns:\n"                                                                       //
    "  int: Hash of all data absorbed so far.\n"                                       //
    "Example:\n"                                                                       //
    "  >>> isinstance(sz.Hasher().update(b'x').digest(), int)\n"                       //
    "  True";

static char const doc_Hasher_hexdigest[] =                              //
    "hexdigest() -> str\n"                                              //
    "\n"                                                                //
    "Return the current hash as a 16-character lowercase hex string.\n" //
    "\n"                                                                //
    "Returns:\n"                                                        //
    "  str: Zero-padded 16-digit hex of the 64-bit hash.\n"             //
    "Example:\n"                                                        //
    "  >>> len(sz.Hasher().hexdigest())\n"                              //
    "  16";

static char const doc_Hasher_reset[] =                       //
    "reset() -> Hasher\n"                                    //
    "\n"                                                     //
    "Reset the state to the initial seed and return self.\n" //
    "\n"                                                     //
    "Returns:\n"                                             //
    "  Hasher: The same object, emptied and reusable.\n"     //
    "Example:\n"                                             //
    "  >>> h = sz.Hasher().update(b'data').reset()\n"        //
    "  >>> h.digest() == sz.Hasher().digest()\n"             //
    "  True";

static PyMethodDef Hasher_methods[] = {
    {"update", (PyCFunction)Hasher_update, METH_O, doc_Hasher_update},               //
    {"digest", (PyCFunction)Hasher_digest, METH_NOARGS, doc_Hasher_digest},          //
    {"hexdigest", (PyCFunction)Hasher_hexdigest, METH_NOARGS, doc_Hasher_hexdigest}, //
    {"reset", (PyCFunction)Hasher_reset, METH_NOARGS, doc_Hasher_reset},             //
    {NULL, NULL, 0, NULL},
};

PyTypeObject HasherType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Hasher",
    .tp_doc = doc_Hasher,
    .tp_basicsize = sizeof(Hasher),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Hasher_new,
    .tp_init = (initproc)Hasher_init,
    .tp_dealloc = (destructor)Hasher_dealloc,
    .tp_methods = Hasher_methods,
};

typedef struct {
    PyObject ob_base;
    sz_sha256_state_t state;
} Sha256;

static void Sha256_dealloc(Sha256 *self) { Py_TYPE(self)->tp_free((PyObject *)self); }

static PyObject *Sha256_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    sz_unused_(args), sz_unused_(kwds);
    Sha256 *self = (Sha256 *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    sz_sha256_state_init(&self->state);
    return (PyObject *)self;
}

static int Sha256_init(Sha256 *self, PyObject *args, PyObject *kwargs) {
    // No arguments expected
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs > 0) {
        PyErr_SetString(PyExc_TypeError, "Sha256() takes no positional arguments");
        return -1;
    }
    if (kwargs && PyDict_Size(kwargs) > 0) {
        PyErr_SetString(PyExc_TypeError, "Sha256() takes no keyword arguments");
        return -1;
    }
    sz_sha256_state_init(&self->state);
    return 0;
}

static PyObject *Sha256_update(PyObject *self_obj, PyObject *arg) {
    Sha256 *self = (Sha256 *)self_obj;
    sz_string_view_t text;
    if (!sz_py_export_string_like(arg, &text.start, &text.length)) {
        wrap_current_exception("Argument must be string-like");
        return NULL;
    }
    sz_sha256_state_update(&self->state, text.start, text.length);
    Py_INCREF(self_obj);
    return self_obj;
}

static PyObject *Sha256_digest(PyObject *self_obj, PyObject *noargs) {
    sz_unused_(noargs);
    Sha256 *self = (Sha256 *)self_obj;
    sz_u8_t digest[32];
    sz_sha256_state_digest(&self->state, digest);
    return PyBytes_FromStringAndSize((char const *)digest, 32);
}

static PyObject *Sha256_hexdigest(PyObject *self_obj, PyObject *noargs) {
    sz_unused_(noargs);
    Sha256 *self = (Sha256 *)self_obj;
    sz_u8_t digest[32];
    sz_sha256_state_digest(&self->state, digest);
    char buf[65]; // 64 hex digits + null terminator
    for (int i = 0; i < 32; ++i) snprintf(buf + i * 2, 3, "%02x", digest[i]);
    return PyUnicode_FromString(buf);
}

static PyObject *Sha256_reset(PyObject *self_obj, PyObject *noargs) {
    sz_unused_(noargs);
    Sha256 *self = (Sha256 *)self_obj;
    sz_sha256_state_init(&self->state);
    Py_INCREF(self_obj);
    return self_obj;
}

static PyObject *Sha256_copy(PyObject *self_obj, PyObject *noargs) {
    sz_unused_(noargs);
    Sha256 *self = (Sha256 *)self_obj;
    Sha256 *copy = (Sha256 *)Sha256_new(&Sha256Type, NULL, NULL);
    if (!copy) return NULL;
    copy->state = self->state;
    return (PyObject *)copy;
}

static char const doc_Sha256[] =                                                             //
    "Sha256()\n"                                                                             //
    "\n"                                                                                     //
    "Incremental SHA-256 hasher with a hashlib-compatible interface, hardware-accelerated\n" //
    "where available. Feed data with `update()`, then read `digest()` or `hexdigest()`.\n"   //
    "\n"                                                                                     //
    "Example:\n"                                                                             //
    "  >>> import hashlib\n"                                                                 //
    "  >>> sz.Sha256().update(b'abc').hexdigest() == hashlib.sha256(b'abc').hexdigest()\n"   //
    "  True";

static char const doc_Sha256_update[] =                                                //
    "update(data) -> Sha256\n"                                                         //
    "\n"                                                                               //
    "Absorb more bytes into the running SHA-256 state and return self for chaining.\n" //
    "\n"                                                                               //
    "Args:\n"                                                                          //
    "  data (str | bytes): Chunk to mix into the state.\n"                             //
    "Returns:\n"                                                                       //
    "  Sha256: The same object, enabling `h.update(a).update(b)`.\n"                   //
    "Example:\n"                                                                       //
    "  >>> sz.Sha256().update(b'a').update(b'bc').hexdigest()[:8]\n"                   //
    "  'ba7816bf'";

static char const doc_Sha256_digest[] =                                    //
    "digest() -> bytes\n"                                                  //
    "\n"                                                                   //
    "Return the current 32-byte SHA-256 digest without consuming state.\n" //
    "\n"                                                                   //
    "Returns:\n"                                                           //
    "  bytes: The 32-byte digest.\n"                                       //
    "Example:\n"                                                           //
    "  >>> len(sz.Sha256().update(b'abc').digest())\n"                     //
    "  32";

static char const doc_Sha256_hexdigest[] =                                                 //
    "hexdigest() -> str\n"                                                                 //
    "\n"                                                                                   //
    "Return the current SHA-256 digest as a 64-character lowercase hex string.\n"          //
    "\n"                                                                                   //
    "Returns:\n"                                                                           //
    "  str: 64-digit hex of the 32-byte digest.\n"                                         //
    "Example:\n"                                                                           //
    "  >>> import hashlib\n"                                                               //
    "  >>> sz.Sha256().update(b'abc').hexdigest() == hashlib.sha256(b'abc').hexdigest()\n" //
    "  True";

static char const doc_Sha256_reset[] =                                    //
    "reset() -> Sha256\n"                                                 //
    "\n"                                                                  //
    "Reset the state to the initial SHA-256 constants and return self.\n" //
    "\n"                                                                  //
    "Returns:\n"                                                          //
    "  Sha256: The same object, emptied and reusable.\n"                  //
    "Example:\n"                                                          //
    "  >>> h = sz.Sha256().update(b'x').reset()\n"                        //
    "  >>> h.hexdigest() == sz.Sha256().hexdigest()\n"                    //
    "  True";

static char const doc_Sha256_copy[] =                                                     //
    "copy() -> Sha256\n"                                                                  //
    "\n"                                                                                  //
    "Return an independent copy of the hasher with identical internal state.\n"           //
    "\n"                                                                                  //
    "Returns:\n"                                                                          //
    "  Sha256: A new object that can be advanced separately from the original.\n"         //
    "Example:\n"                                                                          //
    "  >>> a = sz.Sha256().update(b'ab')\n"                                               //
    "  >>> a.copy().update(b'c').hexdigest() == sz.Sha256().update(b'abc').hexdigest()\n" //
    "  True";

static PyMethodDef Sha256_methods[] = {
    {"update", (PyCFunction)Sha256_update, METH_O, doc_Sha256_update},               //
    {"digest", (PyCFunction)Sha256_digest, METH_NOARGS, doc_Sha256_digest},          //
    {"hexdigest", (PyCFunction)Sha256_hexdigest, METH_NOARGS, doc_Sha256_hexdigest}, //
    {"reset", (PyCFunction)Sha256_reset, METH_NOARGS, doc_Sha256_reset},             //
    {"copy", (PyCFunction)Sha256_copy, METH_NOARGS, doc_Sha256_copy},                //
    {NULL, NULL, 0, NULL},
};

PyTypeObject Sha256Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Sha256",
    .tp_doc = doc_Sha256,
    .tp_basicsize = sizeof(Sha256),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Sha256_new,
    .tp_init = (initproc)Sha256_init,
    .tp_dealloc = (destructor)Sha256_dealloc,
    .tp_methods = Sha256_methods,
};

typedef struct {
    PyObject ob_base;
    sz_sha256_state_t *states;
    sz_string_view_t *chunks; //< Reused across `update` calls, so streaming never allocates
    sz_size_t lanes_count;
} Sha256s;

static void Sha256s_dealloc(Sha256s *self) {
    if (self->states) free(self->states);
    if (self->chunks) free(self->chunks);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Sha256s_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    sz_unused_(args), sz_unused_(kwds);
    Sha256s *self = (Sha256s *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->states = NULL, self->chunks = NULL, self->lanes_count = 0;
    return (PyObject *)self;
}

static int Sha256s_init(Sha256s *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t const positional_args_count = PyTuple_Size(args);
    if (positional_args_count != 1 || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "Sha256s() takes exactly one positional argument: the lane count");
        return -1;
    }
    Py_ssize_t const lanes_count = PyNumber_AsSsize_t(PyTuple_GET_ITEM(args, 0), PyExc_OverflowError);
    if (lanes_count == -1 && PyErr_Occurred()) return -1;
    if (lanes_count < 0) {
        PyErr_SetString(PyExc_ValueError, "The lane count must not be negative");
        return -1;
    }

    if (self->states) free(self->states);
    if (self->chunks) free(self->chunks);
    self->states = NULL, self->chunks = NULL, self->lanes_count = (sz_size_t)lanes_count;
    if (lanes_count) {
        self->states = (sz_sha256_state_t *)malloc((size_t)lanes_count * sizeof(sz_sha256_state_t));
        self->chunks = (sz_string_view_t *)malloc((size_t)lanes_count * sizeof(sz_string_view_t));
        if (!self->states || !self->chunks) {
            PyErr_NoMemory();
            return -1;
        }
        for (Py_ssize_t lane_index = 0; lane_index != lanes_count; ++lane_index)
            sz_sha256_state_init(&self->states[lane_index]);
    }
    return 0;
}

static PyObject *Sha256s_update(PyObject *self_obj, PyObject *arg) {
    Sha256s *self = (Sha256s *)self_obj;
    sz_sequence_t texts;
    PyObject *pin = Sha256_bind_texts_(arg, &texts, self->chunks, self->lanes_count);
    if (!pin) return NULL;
    if (texts.count != self->lanes_count) {
        Py_DECREF(pin);
        PyErr_Format(PyExc_ValueError, "Expected exactly one chunk per lane, got %zu for %zu lanes", texts.count,
                     self->lanes_count);
        return NULL;
    }

    // Every string is already a raw pointer pinned by `pin`, and the kernel touches no Python object.
    if (texts.count) {
        Py_BEGIN_ALLOW_THREADS;
        sz_sha256_multistate_update(self->states, &texts);
        Py_END_ALLOW_THREADS;
    }

    Py_DECREF(pin);
    Py_INCREF(self_obj);
    return self_obj;
}

static PyObject *Sha256s_digest(PyObject *self_obj, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple) {
    Sha256s *self = (Sha256s *)self_obj;
    PyObject *out_obj = NULL;
    if (positional_args_count > 1) {
        PyErr_SetString(PyExc_TypeError, "digest() takes at most one positional argument: the output buffer");
        return NULL;
    }
    if (positional_args_count == 1) out_obj = args[0];

    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t i = 0; i < args_names_count; ++i) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, i);
        PyObject *const value = args[positional_args_count + i];
        if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "digest() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }

    // Caller buffer path: the C layout is already one contiguous run of digests, so a `(lanes, 32)`
    // row-major matrix of bytes receives them with no copy and no scratch.
    if (out_obj && out_obj != Py_None) {
        Py_buffer out_view;
        if (!Sha256_bind_digests_(out_obj, self->lanes_count, &out_view)) return NULL;
        if (self->lanes_count) {
            Py_BEGIN_ALLOW_THREADS;
            sz_sha256_multistate_digest(self->states, self->lanes_count, (sz_u8_t *)out_view.buf);
            Py_END_ALLOW_THREADS;
        }
        PyBuffer_Release(&out_view);
        Py_INCREF(out_obj);
        return out_obj;
    }

    if (!self->lanes_count) return PyList_New(0);
    sz_u8_t *digests = (sz_u8_t *)malloc((size_t)self->lanes_count * SZ_SHA256_DIGEST_LENGTH);
    if (!digests) return PyErr_NoMemory();
    Py_BEGIN_ALLOW_THREADS;
    sz_sha256_multistate_digest(self->states, self->lanes_count, digests);
    Py_END_ALLOW_THREADS;
    PyObject *result = Sha256_digests_to_list_(digests, self->lanes_count);
    free(digests);
    return result;
}

static PyObject *Sha256s_hexdigest(PyObject *self_obj, PyObject *noargs) {
    sz_unused_(noargs);
    Sha256s *self = (Sha256s *)self_obj;
    if (!self->lanes_count) return PyList_New(0);

    sz_u8_t *digests = (sz_u8_t *)malloc((size_t)self->lanes_count * SZ_SHA256_DIGEST_LENGTH);
    if (!digests) return PyErr_NoMemory();
    Py_BEGIN_ALLOW_THREADS;
    sz_sha256_multistate_digest(self->states, self->lanes_count, digests);
    Py_END_ALLOW_THREADS;

    PyObject *result = PyList_New((Py_ssize_t)self->lanes_count);
    if (!result) {
        free(digests);
        return NULL;
    }
    for (sz_size_t lane_index = 0; lane_index != self->lanes_count; ++lane_index) {
        char text[SZ_SHA256_DIGEST_LENGTH * 2 + 1];
        sz_u8_t const *digest = &digests[lane_index * SZ_SHA256_DIGEST_LENGTH];
        for (int byte_index = 0; byte_index < SZ_SHA256_DIGEST_LENGTH; ++byte_index)
            snprintf(text + byte_index * 2, 3, "%02x", digest[byte_index]);
        PyObject *hex = PyUnicode_FromString(text);
        if (!hex) {
            free(digests);
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, (Py_ssize_t)lane_index, hex);
    }
    free(digests);
    return result;
}

static PyObject *Sha256s_reset(PyObject *self_obj, PyObject *noargs) {
    sz_unused_(noargs);
    Sha256s *self = (Sha256s *)self_obj;
    for (sz_size_t lane_index = 0; lane_index != self->lanes_count; ++lane_index)
        sz_sha256_state_init(&self->states[lane_index]);
    Py_INCREF(self_obj);
    return self_obj;
}

static PyObject *Sha256s_copy(PyObject *self_obj, PyObject *noargs) {
    sz_unused_(noargs);
    Sha256s *self = (Sha256s *)self_obj;
    Sha256s *copy = (Sha256s *)Sha256s_new(&Sha256sType, NULL, NULL);
    if (!copy) return NULL;
    copy->lanes_count = self->lanes_count;
    if (self->lanes_count) {
        sz_size_t const states_bytes = self->lanes_count * sizeof(sz_sha256_state_t);
        copy->states = (sz_sha256_state_t *)malloc((size_t)states_bytes);
        copy->chunks = (sz_string_view_t *)malloc((size_t)self->lanes_count * sizeof(sz_string_view_t));
        if (!copy->states || !copy->chunks) {
            Py_DECREF(copy);
            return PyErr_NoMemory();
        }
        sz_copy((sz_ptr_t)copy->states, (sz_cptr_t)self->states, states_bytes);
    }
    return (PyObject *)copy;
}

static Py_ssize_t Sha256s_length(PyObject *self_obj) { return (Py_ssize_t)((Sha256s *)self_obj)->lanes_count; }

static PyObject *Sha256s_lane(PyObject *self_obj, Py_ssize_t lane_index) {
    Sha256s *self = (Sha256s *)self_obj;
    if (lane_index < 0 || (sz_size_t)lane_index >= self->lanes_count) {
        PyErr_SetString(PyExc_IndexError, "Lane index out of range");
        return NULL;
    }
    Sha256 *lane = (Sha256 *)Sha256_new(&Sha256Type, NULL, NULL);
    if (!lane) return NULL;
    lane->state = self->states[lane_index];
    return (PyObject *)lane;
}

/**
 *  @brief Installs @p lane_obj into a lane, so a finished lane can be retired without disturbing its
 *         neighbours - the pattern that lets a fixed pool of lanes stream a longer list of files.
 */
static int Sha256s_set_lane(PyObject *self_obj, Py_ssize_t lane_index, PyObject *lane_obj) {
    Sha256s *self = (Sha256s *)self_obj;
    if (lane_index < 0 || (sz_size_t)lane_index >= self->lanes_count) {
        PyErr_SetString(PyExc_IndexError, "Lane index out of range");
        return -1;
    }
    if (!lane_obj) {
        PyErr_SetString(PyExc_TypeError, "Lanes cannot be deleted, only replaced");
        return -1;
    }
    if (!PyObject_TypeCheck(lane_obj, &Sha256Type)) {
        PyErr_Format(PyExc_TypeError, "Expected a Sha256, got %s", Py_TYPE(lane_obj)->tp_name);
        return -1;
    }

    self->states[lane_index] = ((Sha256 const *)lane_obj)->state;
    return 0;
}

static char const doc_Sha256s[] =                                                            //
    "Sha256s(lanes)\n"                                                                       //
    "\n"                                                                                     //
    "Incremental SHA-256 over many independent messages at once, one message per lane.\n"    //
    "\n"                                                                                     //
    "Hashing a single message is a serial dependency chain no instruction set can speed\n"   //
    "up, but independent messages compress in parallel lanes: sixteen at a time on\n"        //
    "AVX-512, eight on AVX2. Feed at least a few kilobytes per lane per call, as shorter\n"  //
    "chunks never reach the lane-parallel path.\n"                                           //
    "\n"                                                                                     //
    "Lanes are workers rather than messages: assigning `lanes[i] = sz.Sha256()` retires a\n" //
    "finished lane and starts the next message in it, leaving every other lane streaming.\n" //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  lanes (int): How many independent messages to advance together.\n"                    //
    "Example:\n"                                                                             //
    "  >>> lanes = sz.Sha256s(2)\n"                                                          //
    "  >>> _ = lanes.update([b'Hello, ', b'Goodbye, '])\n"                                   //
    "  >>> _ = lanes.update([b'world!', b'world!'])\n"                                       //
    "  >>> lanes.digest()[0] == sz.Sha256().update(b'Hello, world!').digest()\n"             //
    "  True";

static char const doc_Sha256s_update[] =                                                         //
    "update(chunks) -> Sha256s\n"                                                                //
    "\n"                                                                                         //
    "Append the next chunk of every lane's message.\n"                                           //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  chunks (Strs | sequence): Exactly one string-like chunk per lane; may be empty. A Strs\n" //
    "       is read in place, whatever its layout, so a tape never leaves its buffer.\n"         //
    "Returns:\n"                                                                                 //
    "  Sha256s: The same object, to allow chaining.";

static char const doc_Sha256s_digest[] =                                                          //
    "digest(out=None) -> list[bytes] | buffer\n"                                                  //
    "\n"                                                                                          //
    "Return each lane's 32-byte digest, leaving every lane able to accept more data.\n"           //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  out (buffer, optional): Writable, C-contiguous buffer of bytes holding at least\n"         //
    "       `len(lanes) * 32` of them - a `(len(lanes), 32)` numpy.uint8 matrix receives one\n"   //
    "       digest per row with zero allocation. Defaults to None.\n"                             //
    "Returns:\n"                                                                                  //
    "  list[bytes]: One digest per lane in lane order, or `out` itself when a buffer is given.\n" //
    "Example:\n"                                                                                  //
    "  >>> lanes = sz.Sha256s(2)\n"                                                               //
    "  >>> digests = bytearray(len(lanes) * sz.Sha256.digest_length)\n"                           //
    "  >>> _ = lanes.update([b'abc', b'abc']).digest(out=digests)\n"                              //
    "  >>> bytes(digests[:sz.Sha256.digest_length]) == sz.Sha256().update(b'abc').digest()\n"     //
    "  True";

static char const doc_Sha256s_hexdigest[] =                                              //
    "hexdigest() -> list[str]\n"                                                         //
    "\n"                                                                                 //
    "Return each lane's digest as a 64-character lowercase hex string, in lane order.\n" //
    "\n"                                                                                 //
    "Returns:\n"                                                                         //
    "  list[str]: One 64-digit hex string per lane.";

static char const doc_Sha256s_reset[] =                                 //
    "reset() -> Sha256s\n"                                              //
    "\n"                                                                //
    "Return every lane to the initial state, reusing the allocation.\n" //
    "\n"                                                                //
    "Returns:\n"                                                        //
    "  Sha256s: The same object, to allow chaining.";

static char const doc_Sha256s_copy[] =                                         //
    "copy() -> Sha256s\n"                                                      //
    "\n"                                                                       //
    "Return an independent batch whose lanes hold identical internal state.\n" //
    "\n"                                                                       //
    "Returns:\n"                                                               //
    "  Sha256s: A new object that can be advanced separately from the original.";

static PyMethodDef Sha256s_methods[] = {
    {"update", (PyCFunction)Sha256s_update, METH_O, doc_Sha256s_update},               //
    {"digest", (PyCFunction)Sha256s_digest, SZ_METHOD_FLAGS, doc_Sha256s_digest},      //
    {"hexdigest", (PyCFunction)Sha256s_hexdigest, METH_NOARGS, doc_Sha256s_hexdigest}, //
    {"reset", (PyCFunction)Sha256s_reset, METH_NOARGS, doc_Sha256s_reset},             //
    {"copy", (PyCFunction)Sha256s_copy, METH_NOARGS, doc_Sha256s_copy},                //
    {NULL, NULL, 0, NULL},
};

static PySequenceMethods Sha256s_as_sequence = {
    .sq_length = Sha256s_length,
    .sq_item = Sha256s_lane,
    .sq_ass_item = Sha256s_set_lane,
};

PyTypeObject Sha256sType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Sha256s",
    .tp_doc = doc_Sha256s,
    .tp_basicsize = sizeof(Sha256s),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Sha256s_new,
    .tp_init = (initproc)Sha256s_init,
    .tp_dealloc = (destructor)Sha256s_dealloc,
    .tp_methods = Sha256s_methods,
    .tp_as_sequence = &Sha256s_as_sequence,
};
