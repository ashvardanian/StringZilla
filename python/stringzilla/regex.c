/**
 *  @file python/stringzilla/regex.c
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Multi-pattern regular expressions over one compiled vocabulary.
 */
#include "stringzilla.h"

/** A vocabulary compiled once into two automata, streamed over many collections of haystacks. */
typedef struct {
    PyObject ob_base;

    /** Owned; freed in @c tp_dealloc, rebuilt by a second @c __init__. */
    sz_regex_engine_t engine;

    /** Guards the round scratch every call grows and rewrites. */
    sz_py_mutex_t mutex;
} RegexEngine;

#pragma region Construction

/** Reads whether both sides are folded before they meet, or compared byte for byte. */
static int parse_case_sensitivity_(PyObject *sensitivity_obj, sz_case_sensitivity_t *result) {
    if (!sensitivity_obj || sensitivity_obj == Py_None) {
        *result = sz_cased_k;
        return 0;
    }
    if (!PyUnicode_Check(sensitivity_obj)) {
        PyErr_SetString(PyExc_TypeError, "case_sensitivity must be a string");
        return -1;
    }
    if (PyUnicode_CompareWithASCIIString(sensitivity_obj, "cased") == 0) {
        *result = sz_cased_k;
        return 0;
    }
    if (PyUnicode_CompareWithASCIIString(sensitivity_obj, "uncased") == 0) {
        *result = sz_uncased_k;
        return 0;
    }
    PyErr_SetString(PyExc_ValueError, "Unknown case sensitivity, expected 'cased' or 'uncased'");
    return -1;
}

static void RegexEngine_dealloc(RegexEngine *self) {
    sz_regex_engine_free(&self->engine, STRINGZILLA_NULL);
    sz_py_mutex_close_(&self->mutex);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int RegexEngine_init(RegexEngine *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t const positional_count = args ? PyTuple_GET_SIZE(args) : 0;
    if (positional_count < 1 || positional_count > 2) {
        PyErr_Format(PyExc_TypeError, "RegexEngine() takes 1 to 2 positional arguments, got %zd", positional_count);
        return -1;
    }
    PyObject *const patterns_object = PyTuple_GET_ITEM(args, 0);
    PyObject *case_sensitivity_obj = positional_count > 1 ? PyTuple_GET_ITEM(args, 1) : NULL;
    PyObject *capabilities_object = NULL, *stream_object = NULL;
    if (kwargs) {
        Py_ssize_t keyword_cursor = 0;
        PyObject *key = NULL, *value = NULL;
        while (PyDict_Next(kwargs, &keyword_cursor, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "case_sensitivity") == 0) {
                if (case_sensitivity_obj) {
                    PyErr_SetString(PyExc_TypeError,
                                    "RegexEngine() got multiple values for argument 'case_sensitivity'");
                    return -1;
                }
                case_sensitivity_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) { capabilities_object = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) { stream_object = value; }
            else {
                PyErr_Format(PyExc_TypeError, "RegexEngine() got an unexpected keyword argument '%U'", key);
                return -1;
            }
        }
    }

    sz_strings_view_t patterns;
    sz_case_sensitivity_t case_sensitivity;
    sz_capability_t capabilities;
    sz_stream_t stream;
    if (sz_py_export_strings(patterns_object, "patterns", &patterns) != 0) return -1;
    if (parse_case_sensitivity_(case_sensitivity_obj, &case_sensitivity) != 0) return -1;
    if (sz_py_export_engine_placement(capabilities_object, stream_object, &capabilities, &stream) != 0) return -1;
    if (sz_py_mutex_open_(&self->mutex) != 0) return -1;

    sz_size_t malformed_index = 0;
    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    sz_py_mutex_lock_(&self->mutex);
    sz_regex_engine_free(&self->engine, STRINGZILLA_NULL);
    status = sz_regex_engine_init(&self->engine, &patterns, case_sensitivity, &malformed_index, capabilities,
                                  STRINGZILLA_NULL, stream);
    sz_py_mutex_unlock_(&self->mutex);
    Py_END_ALLOW_THREADS;
    if (status == sz_invalid_syntax_k) {
        PyErr_Format(PyExc_ValueError, "RegexEngine() refused pattern %zu, which breaks the supported syntax",
                     (size_t)malformed_index);
        return -1;
    }
    if (status != sz_success_k) {
        sz_py_raise_status(status, "RegexEngine()");
        return -1;
    }
    return 0;
}

#pragma endregion Construction

#pragma region Operations

/** Refuses a verb called on an object whose vocabulary was never compiled. */
static int RegexEngine_ready_(RegexEngine *self) {
    if (self->engine.memory) return 0;
    PyErr_SetString(PyExc_ValueError, "RegexEngine holds no compiled vocabulary");
    return -1;
}

static char const doc_RegexEngine_counts[] =                                                      //
    "counts(haystacks, out, *, stream=None) -> None\n"                                            //
    "\n"                                                                                          //
    "Count the leftmost-first matches of the vocabulary in every haystack, into `out`.\n"         //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  haystacks (Strings): Texts to search, from `Strings.copy` on a GPU.\n"                     //
    "  out (buffer): Writable 1-D buffer of pointer-width unsigned integers, one per haystack.\n" //
    "  stream (int, optional): A stream of the engine's device as an integer, or None.\n"         //
    "Example:\n"                                                                                  //
    "  >>> import struct\n"                                                                       //
    "  >>> engine = sz.RegexEngine(sz.Strings(['[0-9]+', '[a-z]+']))\n"                           //
    "  >>> out = memoryview(bytearray(struct.calcsize('N'))).cast('N')\n"                         //
    "  >>> engine.counts(sz.Strings(['abc 123 de']), out)\n"                                      //
    "  >>> out[0]\n"                                                                              //
    "  3";

static PyObject *RegexEngine_counts(RegexEngine *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple) {
    if (positional_args_count < 1 || positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "counts() takes 1 to 2 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *const haystacks_obj = args[0];
    PyObject *out_obj = positional_args_count > 1 ? args[1] : NULL;
    PyObject *stream_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "out") == 0) {
            if (out_obj) {
                PyErr_SetString(PyExc_TypeError, "counts() got multiple values for argument 'out'");
                return NULL;
            }
            out_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) { stream_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "counts() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (!out_obj) {
        PyErr_SetString(PyExc_TypeError, "counts() needs an `out` buffer to write into");
        return NULL;
    }
    if (RegexEngine_ready_(self) != 0) return NULL;

    sz_capability_t const capability = self->engine.capability;
    sz_strings_view_t haystacks;
    sz_stream_t stream = NULL;
    if (sz_py_export_stream(stream_object, &stream) != 0) return NULL;
    if (!(capability & sz_cap_gpus_k)) stream = NULL;
    if (sz_py_export_strings(haystacks_obj, "haystacks", &haystacks) != 0) return NULL;

    sz_size_t const extents[1] = {haystacks.count};
    sz_size_t strides[1];
    Py_buffer out_view;
    if (sz_py_export_output_buffer(out_obj, "out", (Py_ssize_t)sizeof(sz_size_t), 1, extents, &out_view, strides) != 0)
        return NULL;

    sz_size_t *const counts = (sz_size_t *)out_view.buf;
    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    sz_py_mutex_lock_(&self->mutex);
    // Another thread may have rebuilt the engine since the arguments were bound to its placement.
    if (self->engine.capability != capability) status = sz_unexpected_dimensions_k;
    else status = sz_regex_counts(&self->engine, &haystacks, counts, strides[0], stream);
    sz_py_mutex_unlock_(&self->mutex);
    Py_END_ALLOW_THREADS;
    PyBuffer_Release(&out_view);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "counts()");
        return NULL;
    }
    Py_RETURN_NONE;
}

static char const doc_RegexEngine_find[] =                                                               //
    "find(haystacks, matches, offsets, *, stream=None) -> None\n"                                        //
    "\n"                                                                                                 //
    "Locate every leftmost-first match, writing one row per match and one boundary per haystack.\n"      //
    "\n"                                                                                                 //
    "A capacity too small is not an error: `offsets` is filled either way, which sizes the next call.\n" //
    "\n"                                                                                                 //
    "Args:\n"                                                                                            //
    "  haystacks (Strings): Texts to search, from `Strings.copy` on a GPU.\n"                            //
    "  matches (buffer or None): Writable, fully contiguous 2-D buffer of pointer-width unsigned\n"      //
    "    integers, shaped `(capacity, 4)` - haystack index, pattern index, byte offset, byte length.\n"  //
    "    None makes the call a pure size query.\n"                                                       //
    "  offsets (buffer): Writable, contiguous 1-D buffer of pointer-width unsigned integers holding\n"   //
    "    len(haystacks) + 1 boundaries into `matches`, the last being the total.\n"                      //
    "  stream (int, optional): A stream of the engine's device as an integer, or None.\n"                //
    "Example:\n"                                                                                         //
    "  >>> import struct\n"                                                                              //
    "  >>> engine = sz.RegexEngine(sz.Strings(['[0-9]+']))\n"                                            //
    "  >>> matches = memoryview(bytearray(2 * 4 * struct.calcsize('N'))).cast('N', (2, 4))\n"            //
    "  >>> offsets = memoryview(bytearray(2 * struct.calcsize('N'))).cast('N')\n"                        //
    "  >>> engine.find(sz.Strings(['a 12 b 345']), matches, offsets)\n"                                  //
    "  >>> matches[1, 2], matches[1, 3]\n"                                                               //
    "  (7, 3)";

static PyObject *RegexEngine_find(RegexEngine *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *args_names_tuple) {
    if (positional_args_count < 1 || positional_args_count > 3) {
        PyErr_Format(PyExc_TypeError, "find() takes 1 to 3 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *const haystacks_obj = args[0];
    PyObject *matches_obj = positional_args_count > 1 ? args[1] : NULL;
    PyObject *offsets_obj = positional_args_count > 2 ? args[2] : NULL;
    PyObject *stream_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "matches") == 0) {
            if (matches_obj) {
                PyErr_SetString(PyExc_TypeError, "find() got multiple values for argument 'matches'");
                return NULL;
            }
            matches_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "offsets") == 0) {
            if (offsets_obj) {
                PyErr_SetString(PyExc_TypeError, "find() got multiple values for argument 'offsets'");
                return NULL;
            }
            offsets_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) { stream_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "find() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (!offsets_obj) {
        PyErr_SetString(PyExc_TypeError, "find() needs an `offsets` buffer to write into");
        return NULL;
    }
    if (RegexEngine_ready_(self) != 0) return NULL;

    sz_capability_t const capability = self->engine.capability;
    sz_strings_view_t haystacks;
    sz_stream_t stream = NULL;
    if (sz_py_export_stream(stream_object, &stream) != 0) return NULL;
    if (!(capability & sz_cap_gpus_k)) stream = NULL;
    if (sz_py_export_strings(haystacks_obj, "haystacks", &haystacks) != 0) return NULL;

    // One match is four adjacent pointer-width fields, so rows must be packed as well as columns.
    sz_match_t *matches = STRINGZILLA_NULL;
    sz_size_t matches_capacity = 0;
    Py_buffer matches_view;
    sz_bool_t have_matches = sz_false_k;
    if (matches_obj && matches_obj != Py_None) {
        sz_size_t const matches_extents[2] = {0, 4};
        sz_size_t matches_strides[2];
        if (sz_py_export_output_buffer(matches_obj, "matches", (Py_ssize_t)sizeof(sz_size_t), 2, matches_extents,
                                       &matches_view, matches_strides) != 0)
            return NULL;
        if (matches_strides[0] != 4 || matches_strides[1] != 1) {
            PyErr_SetString(PyExc_ValueError, "matches rows must be contiguous, four fields wide");
            PyBuffer_Release(&matches_view);
            return NULL;
        }
        matches = (sz_match_t *)matches_view.buf;
        matches_capacity = (sz_size_t)matches_view.shape[0];
        have_matches = sz_true_k;
    }

    sz_size_t const offsets_extents[1] = {haystacks.count + 1};
    sz_size_t offsets_strides[1];
    Py_buffer offsets_view;
    if (sz_py_export_output_buffer(offsets_obj, "offsets", (Py_ssize_t)sizeof(sz_size_t), 1, offsets_extents,
                                   &offsets_view, offsets_strides) != 0) {
        if (have_matches) PyBuffer_Release(&matches_view);
        return NULL;
    }
    if (offsets_strides[0] != 1) {
        PyErr_SetString(PyExc_ValueError, "offsets must be contiguous");
        PyBuffer_Release(&offsets_view);
        if (have_matches) PyBuffer_Release(&matches_view);
        return NULL;
    }

    sz_size_t *const offsets = (sz_size_t *)offsets_view.buf;
    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    sz_py_mutex_lock_(&self->mutex);
    // Another thread may have rebuilt the engine since the arguments were bound to its placement.
    if (self->engine.capability != capability) status = sz_unexpected_dimensions_k;
    else status = sz_regex_find(&self->engine, &haystacks, matches, matches_capacity, offsets, stream);
    sz_py_mutex_unlock_(&self->mutex);
    Py_END_ALLOW_THREADS;
    PyBuffer_Release(&offsets_view);
    if (have_matches) PyBuffer_Release(&matches_view);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "find()");
        return NULL;
    }
    Py_RETURN_NONE;
}

static char const doc_RegexEngine_replace[] =                                                          //
    "replace(haystacks, replacements, target, offsets) -> None\n"                                      //
    "\n"                                                                                               //
    "Rewrite every haystack into one target buffer, one replacement per pattern.\n"                    //
    "\n"                                                                                               //
    "A replacement names groups as `$1`, `${1}` or `${name}`, `$0` being the whole match and `$$`\n"   //
    "one dollar. A target too small is not an error: `offsets` is filled either way. CPU only.\n"      //
    "\n"                                                                                               //
    "Args:\n"                                                                                          //
    "  haystacks (Strings): Texts to rewrite.\n"                                                       //
    "  replacements (Strings): One replacement per pattern; an empty one deletes the match.\n"         //
    "  target (buffer or None): Writable, contiguous byte buffer; None makes the call a size query.\n" //
    "  offsets (buffer): Writable, contiguous 1-D buffer of pointer-width unsigned integers holding\n" //
    "    len(haystacks) + 1 rewritten boundaries, the last being the total.\n"                         //
    "Example:\n"                                                                                       //
    "  >>> import struct\n"                                                                            //
    "  >>> engine = sz.RegexEngine(sz.Strings(['(\\\\w+)@(\\\\w+)']))\n"                               //
    "  >>> target = memoryview(bytearray(16))\n"                                                       //
    "  >>> offsets = memoryview(bytearray(2 * struct.calcsize('N'))).cast('N')\n"                      //
    "  >>> engine.replace(sz.Strings(['bob@host']), sz.Strings(['$2/$1']), target, offsets)\n"         //
    "  >>> bytes(target[offsets[0]:offsets[1]])\n"                                                     //
    "  b'host/bob'";

static PyObject *RegexEngine_replace(RegexEngine *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                     PyObject *args_names_tuple) {
    if (positional_args_count < 2 || positional_args_count > 4) {
        PyErr_Format(PyExc_TypeError, "replace() takes 2 to 4 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *const haystacks_obj = args[0];
    PyObject *const replacements_obj = args[1];
    PyObject *target_object = positional_args_count > 2 ? args[2] : NULL;
    PyObject *offsets_obj = positional_args_count > 3 ? args[3] : NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "target") == 0) {
            if (target_object) {
                PyErr_SetString(PyExc_TypeError, "replace() got multiple values for argument 'target'");
                return NULL;
            }
            target_object = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "offsets") == 0) {
            if (offsets_obj) {
                PyErr_SetString(PyExc_TypeError, "replace() got multiple values for argument 'offsets'");
                return NULL;
            }
            offsets_obj = value;
        }
        else {
            PyErr_Format(PyExc_TypeError, "replace() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (!offsets_obj) {
        PyErr_SetString(PyExc_TypeError, "replace() needs an `offsets` buffer to write into");
        return NULL;
    }
    if (RegexEngine_ready_(self) != 0) return NULL;

    sz_strings_view_t haystacks, replacements;
    if (sz_py_export_strings(haystacks_obj, "haystacks", &haystacks) != 0) return NULL;
    if (sz_py_export_strings(replacements_obj, "replacements", &replacements) != 0) return NULL;

    sz_ptr_t target = STRINGZILLA_NULL;
    sz_size_t target_capacity = 0;
    Py_buffer target_view;
    sz_bool_t have_target = sz_false_k;
    if (target_object && target_object != Py_None) {
        sz_size_t const target_extents[1] = {0};
        sz_size_t target_strides[1];
        int const export_status = sz_py_export_output_buffer(target_object, "target", 1, 1, target_extents,
                                                             &target_view, target_strides);
        if (export_status != 0) return NULL;
        if (target_strides[0] != 1) {
            PyErr_SetString(PyExc_ValueError, "target must be contiguous");
            PyBuffer_Release(&target_view);
            return NULL;
        }
        target = (sz_ptr_t)target_view.buf;
        target_capacity = (sz_size_t)target_view.len;
        have_target = sz_true_k;
    }

    sz_size_t const offsets_extents[1] = {haystacks.count + 1};
    sz_size_t offsets_strides[1];
    Py_buffer offsets_view;
    if (sz_py_export_output_buffer(offsets_obj, "offsets", (Py_ssize_t)sizeof(sz_size_t), 1, offsets_extents,
                                   &offsets_view, offsets_strides) != 0) {
        if (have_target) PyBuffer_Release(&target_view);
        return NULL;
    }
    if (offsets_strides[0] != 1) {
        PyErr_SetString(PyExc_ValueError, "offsets must be contiguous");
        PyBuffer_Release(&offsets_view);
        if (have_target) PyBuffer_Release(&target_view);
        return NULL;
    }

    sz_size_t *const offsets = (sz_size_t *)offsets_view.buf;
    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    sz_py_mutex_lock_(&self->mutex);
    status = sz_regex_replace(&self->engine, &haystacks, &replacements, target, target_capacity, offsets,
                              STRINGZILLA_NULL);
    sz_py_mutex_unlock_(&self->mutex);
    Py_END_ALLOW_THREADS;
    PyBuffer_Release(&offsets_view);
    if (have_target) PyBuffer_Release(&target_view);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "replace()");
        return NULL;
    }
    Py_RETURN_NONE;
}

static char const doc_RegexEngine_extract[] =                                                              //
    "extract(haystacks, pattern_index, groups, offsets, *, stream=None) -> None\n"                         //
    "\n"                                                                                                   //
    "Locate the groups of each haystack's first match of a pattern, as Pandas `str.extract` reads them.\n" //
    "\n"                                                                                                   //
    "A haystack with such a match gets a row for the whole match, group index zero, then one row per\n"    //
    "group that took part, so a group left out differs from an empty one; a haystack without one gets\n"   //
    "no rows. A capacity too small is not an error: `offsets` is filled either way, which sizes the\n"     //
    "next call. `haystacks.matched(groups)` reads the rows as strings.\n"                                  //
    "\n"                                                                                                   //
    "Args:\n"                                                                                              //
    "  haystacks (Strings): Texts to search, from `Strings.copy` on a GPU.\n"                              //
    "  pattern_index (int): The pattern whose groups to locate.\n"                                         //
    "  groups (buffer or None): Writable, fully contiguous 2-D buffer of pointer-width unsigned\n"         //
    "    integers, shaped `(capacity, 4)` - haystack index, group index, byte offset, byte length.\n"      //
    "    None makes the call a pure size query.\n"                                                         //
    "  offsets (buffer): Writable, contiguous 1-D buffer of pointer-width unsigned integers holding\n"     //
    "    len(haystacks) + 1 boundaries into `groups`, the last being the total.\n"                         //
    "  stream (int, optional): A stream of the engine's device as an integer, or None.\n"                  //
    "Example:\n"                                                                                           //
    "  >>> import struct\n"                                                                                //
    "  >>> engine = sz.RegexEngine(sz.Strings(['(\\\\d+)-(\\\\d+)']))\n"                                   //
    "  >>> groups = memoryview(bytearray(3 * 4 * struct.calcsize('N'))).cast('N', (3, 4))\n"               //
    "  >>> offsets = memoryview(bytearray(3 * struct.calcsize('N'))).cast('N')\n"                          //
    "  >>> haystacks = sz.Strings(['a 12-34', 'none'])\n"                                                  //
    "  >>> engine.extract(haystacks, 0, groups, offsets)\n"                                                //
    "  >>> list(map(bytes, haystacks.matched(groups))), offsets[2]\n"                                      //
    "  ([b'12-34', b'12', b'34'], 3)";

static PyObject *RegexEngine_extract(RegexEngine *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                     PyObject *args_names_tuple) {
    if (positional_args_count < 2 || positional_args_count > 4) {
        PyErr_Format(PyExc_TypeError, "extract() takes 2 to 4 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *const haystacks_obj = args[0];
    PyObject *groups_object = positional_args_count > 2 ? args[2] : NULL;
    PyObject *offsets_obj = positional_args_count > 3 ? args[3] : NULL;
    PyObject *stream_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "groups") == 0) {
            if (groups_object) {
                PyErr_SetString(PyExc_TypeError, "extract() got multiple values for argument 'groups'");
                return NULL;
            }
            groups_object = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "offsets") == 0) {
            if (offsets_obj) {
                PyErr_SetString(PyExc_TypeError, "extract() got multiple values for argument 'offsets'");
                return NULL;
            }
            offsets_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) { stream_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "extract() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (!offsets_obj) {
        PyErr_SetString(PyExc_TypeError, "extract() needs an `offsets` buffer to write into");
        return NULL;
    }
    if (RegexEngine_ready_(self) != 0) return NULL;
    Py_ssize_t const pattern_index = PyLong_AsSsize_t(args[1]);
    if (pattern_index == -1 && PyErr_Occurred()) return NULL;
    if (pattern_index < 0) {
        PyErr_SetString(PyExc_ValueError, "pattern_index must not be negative");
        return NULL;
    }

    sz_capability_t const capability = self->engine.capability;
    sz_strings_view_t haystacks;
    sz_stream_t stream = NULL;
    if (sz_py_export_stream(stream_object, &stream) != 0) return NULL;
    if (!(capability & sz_cap_gpus_k)) stream = NULL;
    if (sz_py_export_strings(haystacks_obj, "haystacks", &haystacks) != 0) return NULL;

    // One row is four adjacent pointer-width fields, so rows must be packed as well as columns.
    sz_match_t *groups = STRINGZILLA_NULL;
    sz_size_t groups_capacity = 0;
    Py_buffer groups_view;
    sz_bool_t have_groups = sz_false_k;
    if (groups_object && groups_object != Py_None) {
        sz_size_t const groups_extents[2] = {0, 4};
        sz_size_t groups_strides[2];
        if (sz_py_export_output_buffer(groups_object, "groups", (Py_ssize_t)sizeof(sz_size_t), 2, groups_extents,
                                       &groups_view, groups_strides) != 0)
            return NULL;
        if (groups_strides[0] != 4 || groups_strides[1] != 1) {
            PyErr_SetString(PyExc_ValueError, "groups rows must be contiguous, four fields wide");
            PyBuffer_Release(&groups_view);
            return NULL;
        }
        groups = (sz_match_t *)groups_view.buf;
        groups_capacity = (sz_size_t)groups_view.shape[0];
        have_groups = sz_true_k;
    }

    sz_size_t const offsets_extents[1] = {haystacks.count + 1};
    sz_size_t offsets_strides[1];
    Py_buffer offsets_view;
    if (sz_py_export_output_buffer(offsets_obj, "offsets", (Py_ssize_t)sizeof(sz_size_t), 1, offsets_extents,
                                   &offsets_view, offsets_strides) != 0) {
        if (have_groups) PyBuffer_Release(&groups_view);
        return NULL;
    }
    if (offsets_strides[0] != 1) {
        PyErr_SetString(PyExc_ValueError, "offsets must be contiguous");
        PyBuffer_Release(&offsets_view);
        if (have_groups) PyBuffer_Release(&groups_view);
        return NULL;
    }

    sz_size_t *const offsets = (sz_size_t *)offsets_view.buf;
    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    sz_py_mutex_lock_(&self->mutex);
    // Another thread may have rebuilt the engine since the arguments were bound to its placement.
    if (self->engine.capability != capability) status = sz_unexpected_dimensions_k;
    else
        status = sz_regex_extract(&self->engine, &haystacks, (sz_size_t)pattern_index, groups, groups_capacity, offsets,
                                  stream);
    sz_py_mutex_unlock_(&self->mutex);
    Py_END_ALLOW_THREADS;
    PyBuffer_Release(&offsets_view);
    if (have_groups) PyBuffer_Release(&groups_view);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "extract()");
        return NULL;
    }
    Py_RETURN_NONE;
}

#pragma endregion Operations

#pragma region Type Registration

static char const doc_RegexEngine[] =                                                                  //
    "RegexEngine(patterns, case_sensitivity='cased', *, capabilities=None, stream=None)\n"             //
    "\n"                                                                                               //
    "Compile a vocabulary of RE2-syntax patterns once and stream many collections of haystacks\n"      //
    "through it, reporting leftmost-first matches across all of them, the lower index winning ties.\n" //
    "\n"                                                                                               //
    "Args:\n"                                                                                          //
    "  patterns (Strings): The vocabulary to compile; a malformed pattern raises a ValueError\n"       //
    "    naming its index.\n"                                                                          //
    "  case_sensitivity (str, optional): 'cased' matches bytes, 'uncased' folds both sides.\n"         //
    "  capabilities (Capability, optional): A producer's mask, like cuda_capabilities_enabled(0),\n"   //
    "    defaulting to cpu_capabilities_enabled().\n"                                                  //
    "  stream (int, optional): A stream of that vendor as an integer, naming the device, or None\n"    //
    "    for the default.\n"                                                                           //
    "Example:\n"                                                                                       //
    "  >>> import struct\n"                                                                            //
    "  >>> engine = sz.RegexEngine(sz.Strings(['[0-9]+', '[a-z]+']))\n"                                //
    "  >>> out = memoryview(bytearray(struct.calcsize('N'))).cast('N')\n"                              //
    "  >>> engine.counts(sz.Strings(['abc 123 de']), out)\n"                                           //
    "  >>> out[0]\n"                                                                                   //
    "  3";

static PyMethodDef RegexEngine_methods[] = {
    {"counts", (PyCFunction)RegexEngine_counts, STRINGZILLA_METHOD_FLAGS, doc_RegexEngine_counts},
    {"find", (PyCFunction)RegexEngine_find, STRINGZILLA_METHOD_FLAGS, doc_RegexEngine_find},
    {"replace", (PyCFunction)RegexEngine_replace, STRINGZILLA_METHOD_FLAGS, doc_RegexEngine_replace},
    {"extract", (PyCFunction)RegexEngine_extract, STRINGZILLA_METHOD_FLAGS, doc_RegexEngine_extract},
    {NULL, NULL, 0, NULL},
};

PyTypeObject RegexEngineType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.RegexEngine",
    .tp_doc = doc_RegexEngine,
    .tp_basicsize = sizeof(RegexEngine),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)RegexEngine_init,
    .tp_dealloc = (destructor)RegexEngine_dealloc,
    .tp_methods = RegexEngine_methods,
};

#pragma endregion Type Registration
