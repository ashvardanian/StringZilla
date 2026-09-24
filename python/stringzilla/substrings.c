/**
 *  @file python/stringzilla/substrings.c
 *  @author Ash Vardanian
 *  @date August 8, 2026
 *  @brief Multi-pattern substring search over one compiled vocabulary.
 */
#include "stringzilla.h"

/** A vocabulary compiled once into an automaton, streamed over many collections of haystacks. */
typedef struct {
    PyObject ob_base;
    sz_substrings_engine_t engine; //< Owned; freed in `tp_dealloc`, rebuilt by a second `__init__`.
    sz_engine_lock_field_          //< Guards the round arena and the report every call rewrites.
} SubstringsEngine;

#pragma region Construction

/** Reads whether both sides are folded before they meet, or compared byte for byte. */
static int parse_case_sensitivity_(PyObject *sensitivity_obj, sz_substrings_case_sensitivity_t *result) {
    if (!sensitivity_obj || sensitivity_obj == Py_None) {
        *result = sz_substrings_cased_k;
        return 0;
    }
    if (!PyUnicode_Check(sensitivity_obj)) {
        PyErr_SetString(PyExc_TypeError, "case_sensitivity must be a string");
        return -1;
    }
    if (PyUnicode_CompareWithASCIIString(sensitivity_obj, "cased") == 0) {
        *result = sz_substrings_cased_k;
        return 0;
    }
    if (PyUnicode_CompareWithASCIIString(sensitivity_obj, "uncased") == 0) {
        *result = sz_substrings_uncased_k;
        return 0;
    }
    PyErr_SetString(PyExc_ValueError, "Unknown case sensitivity, expected 'cased' or 'uncased'");
    return -1;
}

/** Reads how matches that share bytes resolve, which is also what the round arena is sized for. */
static int parse_overlap_policy_(PyObject *policy_obj, sz_substrings_overlap_policy_t *result) {
    if (!policy_obj || policy_obj == Py_None) {
        *result = sz_substrings_overlapping_k;
        return 0;
    }
    if (!PyUnicode_Check(policy_obj)) {
        PyErr_SetString(PyExc_TypeError, "overlap_policy must be a string");
        return -1;
    }
    if (PyUnicode_CompareWithASCIIString(policy_obj, "overlapping") == 0) {
        *result = sz_substrings_overlapping_k;
        return 0;
    }
    if (PyUnicode_CompareWithASCIIString(policy_obj, "leftmost-longest") == 0) {
        *result = sz_substrings_leftmost_longest_k;
        return 0;
    }
    if (PyUnicode_CompareWithASCIIString(policy_obj, "leftmost-first") == 0) {
        *result = sz_substrings_leftmost_first_k;
        return 0;
    }
    PyErr_SetString(PyExc_ValueError,
                    "Unknown overlap policy, expected 'overlapping', 'leftmost-longest', or 'leftmost-first'");
    return -1;
}

/** Reads a non-negative size, leaving @p result at @p fallback when the argument is absent. */
static int parse_optional_size_(PyObject *size_obj, char const *name, sz_size_t fallback, sz_size_t *result) {
    if (!size_obj || size_obj == Py_None) {
        *result = fallback;
        return 0;
    }
    if (!PyLong_Check(size_obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be an int or None", name);
        return -1;
    }
    *result = (sz_size_t)PyLong_AsSize_t(size_obj);
    if (PyErr_Occurred()) return -1;
    return 0;
}

/** Reads the vocabulary and the four build-time knobs both constructors take. */
static int parse_build_arguments_(PyObject *needles_obj, PyObject *case_sensitivity_obj, PyObject *overlap_policy_obj,
                                  PyObject *hot_states_obj, PyObject *matches_budget_obj, sz_sequence_t *needles,
                                  sz_substrings_case_sensitivity_t *case_sensitivity,
                                  sz_substrings_overlap_policy_t *overlap_policy, sz_size_t *hot_states,
                                  sz_size_t *matches_budget) {
    if (sz_py_export_strings(needles_obj, "needles", needles) != 0) return -1;
    if (parse_case_sensitivity_(case_sensitivity_obj, case_sensitivity) != 0) return -1;
    if (parse_overlap_policy_(overlap_policy_obj, overlap_policy) != 0) return -1;
    if (parse_optional_size_(hot_states_obj, "hot_states", SZ_SUBSTRINGS_HOT_STATES_AUTO, hot_states) != 0) return -1;
    if (parse_optional_size_(matches_budget_obj, "matches_budget", 0, matches_budget) != 0) return -1;
    return 0;
}

static void SubstringsEngine_dealloc(SubstringsEngine *self) {
    sz_substrings_engine_free(&self->engine);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int SubstringsEngine_init(SubstringsEngine *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t const positional_count = args ? PyTuple_GET_SIZE(args) : 0;
    if (positional_count < 1 || positional_count > 5) {
        PyErr_Format(PyExc_TypeError, "SubstringsEngine() takes 1 to 5 positional arguments, got %zd",
                     positional_count);
        return -1;
    }
    PyObject *const needles_obj = PyTuple_GET_ITEM(args, 0);
    PyObject *case_sensitivity_obj = positional_count > 1 ? PyTuple_GET_ITEM(args, 1) : NULL;
    PyObject *overlap_policy_obj = positional_count > 2 ? PyTuple_GET_ITEM(args, 2) : NULL;
    PyObject *hot_states_obj = positional_count > 3 ? PyTuple_GET_ITEM(args, 3) : NULL;
    PyObject *matches_budget_obj = positional_count > 4 ? PyTuple_GET_ITEM(args, 4) : NULL;
    if (kwargs) {
        Py_ssize_t keyword_cursor = 0;
        PyObject *key = NULL, *value = NULL;
        while (PyDict_Next(kwargs, &keyword_cursor, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "case_sensitivity") == 0) {
                if (case_sensitivity_obj) {
                    PyErr_SetString(PyExc_TypeError,
                                    "SubstringsEngine() got multiple values for argument 'case_sensitivity'");
                    return -1;
                }
                case_sensitivity_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "overlap_policy") == 0) {
                if (overlap_policy_obj) {
                    PyErr_SetString(PyExc_TypeError,
                                    "SubstringsEngine() got multiple values for argument 'overlap_policy'");
                    return -1;
                }
                overlap_policy_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "hot_states") == 0) {
                if (hot_states_obj) {
                    PyErr_SetString(PyExc_TypeError,
                                    "SubstringsEngine() got multiple values for argument 'hot_states'");
                    return -1;
                }
                hot_states_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "matches_budget") == 0) {
                if (matches_budget_obj) {
                    PyErr_SetString(PyExc_TypeError,
                                    "SubstringsEngine() got multiple values for argument 'matches_budget'");
                    return -1;
                }
                matches_budget_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "SubstringsEngine() got an unexpected keyword argument '%U'", key);
                return -1;
            }
        }
    }

    sz_sequence_t needles;
    sz_substrings_case_sensitivity_t case_sensitivity;
    sz_substrings_overlap_policy_t overlap_policy;
    sz_size_t hot_states = 0, matches_budget = 0;
    if (parse_build_arguments_(needles_obj, case_sensitivity_obj, overlap_policy_obj, hot_states_obj,
                               matches_budget_obj, &needles, &case_sensitivity, &overlap_policy, &hot_states,
                               &matches_budget) != 0)
        return -1;

    sz_engine_lock_(self);
    sz_substrings_engine_free(&self->engine);
    sz_status_t const status = sz_substrings_engine_init_cpu(&needles, case_sensitivity, overlap_policy, hot_states,
                                                             matches_budget, SZ_NULL, &self->engine);
    sz_engine_unlock_(self);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "SubstringsEngine()");
        return -1;
    }
    return 0;
}

static char const doc_SubstringsEngine_on_gpu[] =                                                       //
    "on_gpu(needles, case_sensitivity='cased', overlap_policy='overlapping', hot_states=None,\n"        //
    "       matches_budget=0, stream=0) -> SubstringsEngine\n"                                          //
    "\n"                                                                                                //
    "Compile the same vocabulary where a kernel can reach it, on a stream the caller owns.\n"           //
    "\n"                                                                                                //
    "Args:\n"                                                                                           //
    "  needles (Strs): The vocabulary to compile.\n"                                                    //
    "  case_sensitivity (str, optional): 'cased' or 'uncased'.\n"                                       //
    "  overlap_policy (str, optional): 'overlapping', 'leftmost-longest', or 'leftmost-first'.\n"       //
    "  hot_states (int, optional): States kept in the dense hot rows; None sizes them automatically.\n" //
    "  matches_budget (int, optional): Matches one round may emit; 0 asks for a tier-chosen default.\n" //
    "  stream (int, optional): A `cudaStream_t` as an integer, or 0 for the default stream.\n"          //
    "Returns:\n"                                                                                        //
    "  SubstringsEngine: A device engine whose rounds enqueue and return; join before reading.\n"       //
    "Example:\n"                                                                                        //
    "  >>> engine = sz.SubstringsEngine.on_gpu(sz.Strs(['he', 'she']))  # doctest: +SKIP";

static PyObject *SubstringsEngine_on_gpu(PyObject *type_obj, PyObject *const *args, Py_ssize_t positional_args_count,
                                         PyObject *args_names_tuple) {
    if (positional_args_count < 1 || positional_args_count > 6) {
        PyErr_Format(PyExc_TypeError, "on_gpu() takes 1 to 6 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *const needles_obj = args[0];
    PyObject *case_sensitivity_obj = positional_args_count > 1 ? args[1] : NULL;
    PyObject *overlap_policy_obj = positional_args_count > 2 ? args[2] : NULL;
    PyObject *hot_states_obj = positional_args_count > 3 ? args[3] : NULL;
    PyObject *matches_budget_obj = positional_args_count > 4 ? args[4] : NULL;
    PyObject *stream_obj = positional_args_count > 5 ? args[5] : NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "case_sensitivity") == 0) {
            if (case_sensitivity_obj) {
                PyErr_SetString(PyExc_TypeError, "on_gpu() got multiple values for argument 'case_sensitivity'");
                return NULL;
            }
            case_sensitivity_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "overlap_policy") == 0) {
            if (overlap_policy_obj) {
                PyErr_SetString(PyExc_TypeError, "on_gpu() got multiple values for argument 'overlap_policy'");
                return NULL;
            }
            overlap_policy_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "hot_states") == 0) {
            if (hot_states_obj) {
                PyErr_SetString(PyExc_TypeError, "on_gpu() got multiple values for argument 'hot_states'");
                return NULL;
            }
            hot_states_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "matches_budget") == 0) {
            if (matches_budget_obj) {
                PyErr_SetString(PyExc_TypeError, "on_gpu() got multiple values for argument 'matches_budget'");
                return NULL;
            }
            matches_budget_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) {
            if (stream_obj) {
                PyErr_SetString(PyExc_TypeError, "on_gpu() got multiple values for argument 'stream'");
                return NULL;
            }
            stream_obj = value;
        }
        else {
            PyErr_Format(PyExc_TypeError, "on_gpu() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }

    sz_sequence_t needles;
    sz_substrings_case_sensitivity_t case_sensitivity;
    sz_substrings_overlap_policy_t overlap_policy;
    sz_size_t hot_states = 0, matches_budget = 0;
    void *stream = NULL;
    if (parse_build_arguments_(needles_obj, case_sensitivity_obj, overlap_policy_obj, hot_states_obj,
                               matches_budget_obj, &needles, &case_sensitivity, &overlap_policy, &hot_states,
                               &matches_budget) != 0)
        return NULL;
    if (sz_py_export_stream(stream_obj, &stream) != 0) return NULL;

    PyTypeObject *const type = (PyTypeObject *)type_obj;
    SubstringsEngine *const self = (SubstringsEngine *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    sz_status_t const status = sz_substrings_engine_init_gpu(&needles, case_sensitivity, overlap_policy, hot_states,
                                                             matches_budget, SZ_NULL, stream, &self->engine);
    if (status != sz_success_k) {
        Py_DECREF(self);
        sz_py_raise_status(status, "SubstringsEngine.on_gpu()");
        return NULL;
    }
    return (PyObject *)self;
}

#pragma endregion Construction

#pragma region Operations

/** Refuses a verb called on an object whose vocabulary was never compiled. */
static int SubstringsEngine_ready_(SubstringsEngine *self) {
    if (self->engine.memory) return 0;
    PyErr_SetString(PyExc_ValueError, "SubstringsEngine holds no compiled vocabulary");
    return -1;
}

static char const doc_SubstringsEngine_counts[] =                                                 //
    "counts(haystacks, out) -> None\n"                                                            //
    "\n"                                                                                          //
    "Count the matches of every needle in every haystack, into `out`.\n"                          //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  haystacks (Strs): Texts to search.\n"                                                      //
    "  out (buffer): Writable 1-D buffer of pointer-width unsigned integers, one per haystack.\n" //
    "Example:\n"                                                                                  //
    "  >>> engine = sz.SubstringsEngine(sz.Strs(['he', 'she']))\n"                                //
    "  >>> out = memoryview(bytearray(8)).cast('Q')\n"                                            //
    "  >>> engine.counts(sz.Strs(['hershey']), out)\n"                                            //
    "  >>> out[0]\n"                                                                              //
    "  3";

static PyObject *SubstringsEngine_counts(SubstringsEngine *self, PyObject *const *args,
                                         Py_ssize_t positional_args_count, PyObject *args_names_tuple) {
    if (positional_args_count < 1 || positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "counts() takes 1 to 2 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *const haystacks_obj = args[0];
    PyObject *out_obj = positional_args_count > 1 ? args[1] : NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "out") != 0) {
            PyErr_Format(PyExc_TypeError, "counts() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
        if (out_obj) {
            PyErr_SetString(PyExc_TypeError, "counts() got multiple values for argument 'out'");
            return NULL;
        }
        out_obj = value;
    }
    if (!out_obj) {
        PyErr_SetString(PyExc_TypeError, "counts() needs an `out` buffer to write into");
        return NULL;
    }
    if (SubstringsEngine_ready_(self) != 0) return NULL;

    sz_sequence_t haystacks;
    if (sz_py_export_strings(haystacks_obj, "haystacks", &haystacks) != 0) return NULL;

    sz_size_t const extents[1] = {haystacks.count};
    sz_size_t strides[1];
    Py_buffer out_view;
    if (sz_py_export_output_buffer(out_obj, "out", (Py_ssize_t)sizeof(sz_size_t), 1, extents, &out_view, strides) != 0)
        return NULL;

    sz_engine_lock_(self);
    sz_status_t const status = sz_substrings_counts(&self->engine, &haystacks, (sz_size_t *)out_view.buf, strides[0]);
    sz_engine_unlock_(self);
    PyBuffer_Release(&out_view);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "counts()");
        return NULL;
    }
    Py_RETURN_NONE;
}

static char const doc_SubstringsEngine_find[] =                                                        //
    "find(haystacks, matches, offsets) -> None\n"                                                      //
    "\n"                                                                                               //
    "Locate every match, writing one row per match and one boundary per haystack.\n"                   //
    "\n"                                                                                               //
    "A capacity too small is not an error: `offsets` and `report` are filled either way, which is\n"   //
    "what sizes the next call.\n"                                                                      //
    "\n"                                                                                               //
    "Args:\n"                                                                                          //
    "  haystacks (Strs): Texts to search.\n"                                                           //
    "  matches (buffer or None): Writable, fully contiguous 2-D buffer of pointer-width unsigned\n"    //
    "    integers, shaped `(capacity, 4)` - haystack index, needle index, byte offset, byte length.\n" //
    "    None makes the call a pure size query.\n"                                                     //
    "  offsets (buffer): Writable, contiguous 1-D buffer of pointer-width unsigned integers holding\n" //
    "    len(haystacks) + 1 boundaries into `matches`, the last being the total.\n"                    //
    "Example:\n"                                                                                       //
    "  >>> engine = sz.SubstringsEngine(sz.Strs(['he', 'she']))\n"                                     //
    "  >>> matches = memoryview(bytearray(3 * 4 * 8)).cast('Q', (3, 4))\n"                             //
    "  >>> offsets = memoryview(bytearray(2 * 8)).cast('Q')\n"                                         //
    "  >>> engine.find(sz.Strs(['hershey']), matches, offsets)\n"                                      //
    "  >>> offsets[1]\n"                                                                               //
    "  3";

static PyObject *SubstringsEngine_find(SubstringsEngine *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                       PyObject *args_names_tuple) {
    if (positional_args_count < 1 || positional_args_count > 3) {
        PyErr_Format(PyExc_TypeError, "find() takes 1 to 3 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *const haystacks_obj = args[0];
    PyObject *matches_obj = positional_args_count > 1 ? args[1] : NULL;
    PyObject *offsets_obj = positional_args_count > 2 ? args[2] : NULL;
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
        else {
            PyErr_Format(PyExc_TypeError, "find() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (!offsets_obj) {
        PyErr_SetString(PyExc_TypeError, "find() needs an `offsets` buffer to write into");
        return NULL;
    }
    if (SubstringsEngine_ready_(self) != 0) return NULL;

    sz_sequence_t haystacks;
    if (sz_py_export_strings(haystacks_obj, "haystacks", &haystacks) != 0) return NULL;

    // One match is four pointer-width fields back to back, so the rows have to be packed as well as the columns.
    sz_substrings_match_t *matches = SZ_NULL;
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
        matches = (sz_substrings_match_t *)matches_view.buf;
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

    sz_engine_lock_(self);
    sz_status_t const status = sz_substrings_find(&self->engine, &haystacks, matches, matches_capacity,
                                                  (sz_size_t *)offsets_view.buf);
    sz_engine_unlock_(self);
    PyBuffer_Release(&offsets_view);
    if (have_matches) PyBuffer_Release(&matches_view);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "find()");
        return NULL;
    }
    Py_RETURN_NONE;
}

static char const doc_SubstringsEngine_replace[] =                                                     //
    "replace(haystacks, replacements, tape, offsets) -> None\n"                                        //
    "\n"                                                                                               //
    "Rewrite every haystack onto one output tape, one replacement per needle.\n"                       //
    "\n"                                                                                               //
    "Needs a leftmost policy, since a substitution over matches that share bytes is not a function.\n" //
    "A tape too small is not an error: `offsets` and `report` are filled either way.\n"                //
    "\n"                                                                                               //
    "Args:\n"                                                                                          //
    "  haystacks (Strs): Texts to rewrite.\n"                                                          //
    "  replacements (Strs): One replacement per needle; an empty one deletes the match.\n"             //
    "  tape (buffer or None): Writable, contiguous byte buffer; None makes the call a size query.\n"   //
    "  offsets (buffer): Writable, contiguous 1-D buffer of pointer-width unsigned integers holding\n" //
    "    len(haystacks) + 1 rewritten boundaries, the last being the total.\n"                         //
    "Example:\n"                                                                                       //
    "  >>> engine = sz.SubstringsEngine(sz.Strs(['he', 'she']), overlap_policy='leftmost-longest')\n"  //
    "  >>> tape = memoryview(bytearray(16))\n"                                                         //
    "  >>> offsets = memoryview(bytearray(2 * 8)).cast('Q')\n"                                         //
    "  >>> engine.replace(sz.Strs(['hershey']), sz.Strs(['HE', 'SHE']), tape, offsets)\n"              //
    "  >>> bytes(tape[offsets[0]:offsets[1]])\n"                                                       //
    "  b'HErSHEy'";

static PyObject *SubstringsEngine_replace(SubstringsEngine *self, PyObject *const *args,
                                          Py_ssize_t positional_args_count, PyObject *args_names_tuple) {
    if (positional_args_count < 2 || positional_args_count > 4) {
        PyErr_Format(PyExc_TypeError, "replace() takes 2 to 4 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *const haystacks_obj = args[0];
    PyObject *const replacements_obj = args[1];
    PyObject *tape_obj = positional_args_count > 2 ? args[2] : NULL;
    PyObject *offsets_obj = positional_args_count > 3 ? args[3] : NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "tape") == 0) {
            if (tape_obj) {
                PyErr_SetString(PyExc_TypeError, "replace() got multiple values for argument 'tape'");
                return NULL;
            }
            tape_obj = value;
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
    if (SubstringsEngine_ready_(self) != 0) return NULL;

    sz_sequence_t haystacks, replacements;
    if (sz_py_export_strings(haystacks_obj, "haystacks", &haystacks) != 0) return NULL;
    if (sz_py_export_strings(replacements_obj, "replacements", &replacements) != 0) return NULL;

    sz_ptr_t tape = SZ_NULL;
    sz_size_t tape_capacity = 0;
    Py_buffer tape_view;
    sz_bool_t have_tape = sz_false_k;
    if (tape_obj && tape_obj != Py_None) {
        sz_size_t const tape_extents[1] = {0};
        sz_size_t tape_strides[1];
        if (sz_py_export_output_buffer(tape_obj, "tape", 1, 1, tape_extents, &tape_view, tape_strides) != 0)
            return NULL;
        if (tape_strides[0] != 1) {
            PyErr_SetString(PyExc_ValueError, "tape must be contiguous");
            PyBuffer_Release(&tape_view);
            return NULL;
        }
        tape = (sz_ptr_t)tape_view.buf;
        tape_capacity = (sz_size_t)tape_view.len;
        have_tape = sz_true_k;
    }

    sz_size_t const offsets_extents[1] = {haystacks.count + 1};
    sz_size_t offsets_strides[1];
    Py_buffer offsets_view;
    if (sz_py_export_output_buffer(offsets_obj, "offsets", (Py_ssize_t)sizeof(sz_size_t), 1, offsets_extents,
                                   &offsets_view, offsets_strides) != 0) {
        if (have_tape) PyBuffer_Release(&tape_view);
        return NULL;
    }
    if (offsets_strides[0] != 1) {
        PyErr_SetString(PyExc_ValueError, "offsets must be contiguous");
        PyBuffer_Release(&offsets_view);
        if (have_tape) PyBuffer_Release(&tape_view);
        return NULL;
    }

    sz_engine_lock_(self);
    sz_status_t const status = sz_substrings_replace(&self->engine, &haystacks, &replacements, tape, tape_capacity,
                                                     (sz_size_t *)offsets_view.buf);
    sz_engine_unlock_(self);
    PyBuffer_Release(&offsets_view);
    if (have_tape) PyBuffer_Release(&tape_view);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "replace()");
        return NULL;
    }
    Py_RETURN_NONE;
}

static char const doc_SubstringsEngine_bm25_scores[] =                                                  //
    "bm25_scores(haystacks, needle_weights, out, *, document_lengths=None,\n"                           //
    "            term_frequency_saturation=1.2, length_normalization=0.75,\n"                           //
    "            average_document_length=0.0) -> None\n"                                                //
    "\n"                                                                                                //
    "Score every haystack against the vocabulary, which is the query, into `out`.\n"                    //
    "\n"                                                                                                //
    "Term frequencies are raw overlapping counts, so the engine's own overlap policy does not apply.\n" //
    "\n"                                                                                                //
    "Args:\n"                                                                                           //
    "  haystacks (Strs): Documents to score.\n"                                                         //
    "  needle_weights (buffer): One 32-bit float IDF or boost per needle.\n"                            //
    "  out (buffer): Writable 1-D buffer of 32-bit floats, one score per haystack.\n"                   //
    "  document_lengths (buffer, optional): One 32-bit float length per haystack; byte lengths\n"       //
    "    when None.\n"                                                                                  //
    "  term_frequency_saturation (float, optional): The literature's k1.\n"                             //
    "  length_normalization (float, optional): The literature's b, in [0, 1].\n"                        //
    "  average_document_length (float, optional): Corpus-wide mean of the lengths above; required\n"    //
    "    whenever length_normalization is positive.\n"                                                  //
    "Example:\n"                                                                                        //
    "  >>> engine = sz.SubstringsEngine(sz.Strs(['cat', 'dog']))\n"                                     //
    "  >>> weights = memoryview(bytearray(2 * 4)).cast('f')\n"                                          //
    "  >>> weights[0], weights[1] = 1.0, 1.0\n"                                                         //
    "  >>> out = memoryview(bytearray(2 * 4)).cast('f')\n"                                              //
    "  >>> engine.bm25_scores(sz.Strs(['cat and dog', 'nothing here']), weights, out,\n"                //
    "  ...                    average_document_length=10.0)\n"                                          //
    "  >>> out[1]\n"                                                                                    //
    "  0.0";

static PyObject *SubstringsEngine_bm25_scores(SubstringsEngine *self, PyObject *const *args,
                                              Py_ssize_t positional_args_count, PyObject *args_names_tuple) {
    if (positional_args_count < 2 || positional_args_count > 3) {
        PyErr_Format(PyExc_TypeError, "bm25_scores() takes 2 to 3 positional arguments, got %zd",
                     positional_args_count);
        return NULL;
    }
    PyObject *const haystacks_obj = args[0];
    PyObject *const weights_obj = args[1];
    PyObject *out_obj = positional_args_count > 2 ? args[2] : NULL;
    PyObject *lengths_obj = NULL, *saturation_obj = NULL, *normalization_obj = NULL, *average_obj = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "out") == 0) {
            if (out_obj) {
                PyErr_SetString(PyExc_TypeError, "bm25_scores() got multiple values for argument 'out'");
                return NULL;
            }
            out_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "document_lengths") == 0) { lengths_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "term_frequency_saturation") == 0) { saturation_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "length_normalization") == 0) { normalization_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "average_document_length") == 0) { average_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "bm25_scores() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (!out_obj) {
        PyErr_SetString(PyExc_TypeError, "bm25_scores() needs an `out` buffer to write into");
        return NULL;
    }
    if (SubstringsEngine_ready_(self) != 0) return NULL;

    sz_substrings_bm25_t parameters;
    parameters.term_frequency_saturation = 1.2f;
    parameters.length_normalization = 0.75f;
    parameters.average_document_length = 0.0f;
    if (saturation_obj) {
        double const value = PyFloat_AsDouble(saturation_obj);
        if (PyErr_Occurred()) return NULL;
        parameters.term_frequency_saturation = (sz_f32_t)value;
    }
    if (normalization_obj) {
        double const value = PyFloat_AsDouble(normalization_obj);
        if (PyErr_Occurred()) return NULL;
        parameters.length_normalization = (sz_f32_t)value;
    }
    if (average_obj) {
        double const value = PyFloat_AsDouble(average_obj);
        if (PyErr_Occurred()) return NULL;
        parameters.average_document_length = (sz_f32_t)value;
    }

    sz_sequence_t haystacks;
    if (sz_py_export_strings(haystacks_obj, "haystacks", &haystacks) != 0) return NULL;

    Py_buffer weights_view;
    if (sz_py_export_input_buffer(weights_obj, "needle_weights", (Py_ssize_t)sizeof(sz_f32_t),
                                  self->engine.needles_count, &weights_view) != 0)
        return NULL;

    sz_f32_t const *lengths = SZ_NULL;
    Py_buffer lengths_view;
    sz_bool_t have_lengths = sz_false_k;
    if (lengths_obj && lengths_obj != Py_None) {
        if (sz_py_export_input_buffer(lengths_obj, "document_lengths", (Py_ssize_t)sizeof(sz_f32_t), haystacks.count,
                                      &lengths_view) != 0) {
            PyBuffer_Release(&weights_view);
            return NULL;
        }
        lengths = (sz_f32_t const *)lengths_view.buf;
        have_lengths = sz_true_k;
    }

    sz_size_t const extents[1] = {haystacks.count};
    sz_size_t strides[1];
    Py_buffer out_view;
    if (sz_py_export_output_buffer(out_obj, "out", (Py_ssize_t)sizeof(sz_f32_t), 1, extents, &out_view, strides) != 0) {
        if (have_lengths) PyBuffer_Release(&lengths_view);
        PyBuffer_Release(&weights_view);
        return NULL;
    }

    sz_engine_lock_(self);
    sz_status_t const status = sz_substrings_bm25_scores(&self->engine, &haystacks, lengths, &parameters,
                                                         (sz_f32_t const *)weights_view.buf, (sz_f32_t *)out_view.buf,
                                                         strides[0]);
    sz_engine_unlock_(self);
    PyBuffer_Release(&out_view);
    if (have_lengths) PyBuffer_Release(&lengths_view);
    PyBuffer_Release(&weights_view);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "bm25_scores()");
        return NULL;
    }
    Py_RETURN_NONE;
}

static char const doc_SubstringsEngine_report[] =                                                 //
    "What the last round found, as a dict of four counts.\n"                                      //
    "\n"                                                                                          //
    "`matches_emitted` is the truth whatever the outputs could hold, `matches_stored` what was\n" //
    "written, `tape_bytes` what a rewrite needs, and `shortfall` what did not fit. On a device\n" //
    "engine these are written by the device, so join the stream before reading them.\n"           //
    "\n"                                                                                          //
    "Example:\n"                                                                                  //
    "  >>> engine = sz.SubstringsEngine(sz.Strs(['he', 'she']))\n"                                //
    "  >>> out = memoryview(bytearray(8)).cast('Q')\n"                                            //
    "  >>> engine.counts(sz.Strs(['hershey']), out)\n"                                            //
    "  >>> engine.report['matches_emitted']\n"                                                    //
    "  3";

static PyObject *SubstringsEngine_get_report(SubstringsEngine *self, void *closure) {
    sz_unused_(closure);
    sz_substrings_report_t const *const report = self->engine.report;
    if (!report) {
        PyErr_SetString(PyExc_ValueError, "SubstringsEngine holds no compiled vocabulary");
        return NULL;
    }
    return Py_BuildValue("{s:K,s:K,s:K,s:K}",                                            //
                         "matches_emitted", (unsigned long long)report->matches_emitted, //
                         "matches_stored", (unsigned long long)report->matches_stored,   //
                         "tape_bytes", (unsigned long long)report->tape_bytes,           //
                         "shortfall", (unsigned long long)report->shortfall);
}

#pragma endregion Operations

#pragma region Type Registration

static char const doc_SubstringsEngine[] =                                                              //
    "SubstringsEngine(needles, case_sensitivity='cased', overlap_policy='overlapping',\n"               //
    "                 hot_states=None, matches_budget=0)\n"                                             //
    "\n"                                                                                                //
    "Compile a vocabulary once and stream many collections of haystacks through it.\n"                  //
    "\n"                                                                                                //
    "One Aho-Corasick pass answers every needle at once, whatever the vocabulary size.\n"               //
    "\n"                                                                                                //
    "Args:\n"                                                                                           //
    "  needles (Strs): The vocabulary to compile; an empty needle is refused.\n"                        //
    "  case_sensitivity (str, optional): 'cased' matches bytes, 'uncased' folds both sides.\n"          //
    "  overlap_policy (str, optional): 'overlapping', 'leftmost-longest', or 'leftmost-first'.\n"       //
    "  hot_states (int, optional): States kept in the dense hot rows; None sizes them automatically.\n" //
    "  matches_budget (int, optional): Matches one round may emit, read by a device tier only.\n"       //
    "Example:\n"                                                                                        //
    "  >>> engine = sz.SubstringsEngine(sz.Strs(['he', 'she']), case_sensitivity='cased')\n"            //
    "  >>> out = memoryview(bytearray(8)).cast('Q')\n"                                                  //
    "  >>> engine.counts(sz.Strs(['hershey']), out)\n"                                                  //
    "  >>> out[0]\n"                                                                                    //
    "  3";

static PyMethodDef SubstringsEngine_methods[] = {
    {"on_gpu", (PyCFunction)SubstringsEngine_on_gpu, SZ_METHOD_FLAGS | METH_CLASS, doc_SubstringsEngine_on_gpu},
    {"counts", (PyCFunction)SubstringsEngine_counts, SZ_METHOD_FLAGS, doc_SubstringsEngine_counts},
    {"find", (PyCFunction)SubstringsEngine_find, SZ_METHOD_FLAGS, doc_SubstringsEngine_find},
    {"replace", (PyCFunction)SubstringsEngine_replace, SZ_METHOD_FLAGS, doc_SubstringsEngine_replace},
    {"bm25_scores", (PyCFunction)SubstringsEngine_bm25_scores, SZ_METHOD_FLAGS, doc_SubstringsEngine_bm25_scores},
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef SubstringsEngine_getsetters[] = {
    {"report", (getter)SubstringsEngine_get_report, NULL, doc_SubstringsEngine_report, NULL},
    {NULL},
};

PyTypeObject SubstringsEngineType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.SubstringsEngine",
    .tp_doc = doc_SubstringsEngine,
    .tp_basicsize = sizeof(SubstringsEngine),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)SubstringsEngine_init,
    .tp_dealloc = (destructor)SubstringsEngine_dealloc,
    .tp_methods = SubstringsEngine_methods,
    .tp_getset = SubstringsEngine_getsetters,
};

#pragma endregion Type Registration
