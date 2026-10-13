/**
 *  @file python/stringzilla/strings.c
 *  @author Ash Vardanian
 *  @date October 11, 2024
 *  @brief Lookups, escaping, normalization, and JSON and CSV dumps over collections of strings.
 *
 *  Every @c Strs verb here is one call over the whole collection into a new tape, and the dumpers
 *  escape each column once, then interleave the rows straight into one @c bytes object.
 */
#include "stringzilla.h"

sz_status_t sz_py_lookup(sz_ptr_t target, sz_cptr_t source, sz_size_t length, char const *lut,
                         sz_capability_t capabilities) {
    // A one-string tape over the caller's buffers, in the offsets width the kernel writes, so it
    // allocates nothing; in place, the offsets are shared too, making it the very tape it reads.
    sz_u32_t narrow_source[2] = {0, (sz_u32_t)length}, narrow_target[2] = {0, (sz_u32_t)length};
    sz_u64_t wide_source[2] = {0, length}, wide_target[2] = {0, length};
    sz_bool_t const in_place = (sz_bool_t)(source == target);
    sz_strings_view_t strings;
    sz_strings_span_t tape;
    strings.count = tape.count = 1;
    if (length <= 0xFFFFFFFFull) {
        strings.layout = tape.layout = sz_strings_u32_tape_k;
        strings.data.u32_tape.data = source, strings.data.u32_tape.offsets = narrow_source;
        tape.data.u32_tape.data = target, tape.data.u32_tape.offsets = in_place ? narrow_source : narrow_target;
    }
    else {
        strings.layout = tape.layout = sz_strings_u64_tape_k;
        strings.data.u64_tape.data = source, strings.data.u64_tape.offsets = wide_source;
        tape.data.u64_tape.data = target, tape.data.u64_tape.offsets = in_place ? wide_source : wide_target;
    }
    return sz_strings_lookup_best(&strings, NULL, 1, lut, NULL, &tape, capabilities, NULL);
}

/** Reads a 256-byte string or a dict of characters below U+0100, one byte each, into @p lut. */
static int sz_py_export_lookup_table_(PyObject *table_object, char *lut) {
    sz_string_view_t table;
    if (PyDict_Check(table_object)) {
        // Every byte the dict leaves out maps to itself
        for (int i = 0; i < 256; i++) lut[i] = (char)i;
        PyObject *key, *value;
        Py_ssize_t position = 0;
        while (PyDict_Next(table_object, &position, &key, &value)) {
            if (!PyUnicode_Check(key) || PyUnicode_GetLength(key) != 1 || !PyUnicode_Check(value) ||
                PyUnicode_GetLength(value) != 1) {
                PyErr_SetString(PyExc_TypeError, "Keys and values must be single characters");
                return -1;
            }
            Py_UCS4 const key_char = PyUnicode_READ_CHAR(key, 0), value_char = PyUnicode_READ_CHAR(value, 0);
            if (key_char > 255 || value_char > 255) {
                PyErr_SetString(PyExc_ValueError, "Keys and values must be characters below U+0100, naming one byte");
                return -1;
            }
            lut[key_char] = (char)value_char;
        }
        return 0;
    }
    if (!sz_py_export_string_like(table_object, &table.start, &table.length)) {
        wrap_current_exception("The look-up table must be string-like or a dictionary");
        return -1;
    }
    if (table.length != 256) {
        PyErr_SetString(PyExc_ValueError, "The look-up table must be exactly 256 bytes long");
        return -1;
    }
    memcpy(lut, table.start, 256);
    return 0;
}

/** Wraps the tape a verb wrote into a new @c Strs, or frees it and raises the failed status. */
static PyObject *sz_py_strings_result_(sz_status_t status, sz_strings_span_t *tape, sz_allocator_t *allocator,
                                       char const *context) {
    if (status == sz_success_k) return Strs_from_tape_(tape, allocator);
    sz_strings_span_free(tape, allocator, NULL);
    sz_py_raise_status(status, context);
    return NULL;
}

/** The view of string @p index of a tape. */
static sz_string_view_t sz_py_tape_string_(sz_strings_span_t const *tape, sz_size_t index) {
    sz_string_view_t string;
    if (tape->layout == sz_strings_u64_tape_k) {
        sz_u64_t const *offsets = tape->data.u64_tape.offsets + index;
        string.start = tape->data.u64_tape.data + offsets[0], string.length = (sz_size_t)(offsets[1] - offsets[0]);
    }
    else {
        sz_u32_t const *offsets = tape->data.u32_tape.offsets + index;
        string.start = tape->data.u32_tape.data + offsets[0], string.length = offsets[1] - offsets[0];
    }
    return string;
}

/** Bytes in the payload of a tape a verb wrote, which starts at offset zero. */
static sz_size_t sz_py_tape_payload_(sz_strings_span_t const *tape) {
    return tape->layout == sz_strings_u64_tape_k ? (sz_size_t)tape->data.u64_tape.offsets[tape->count]
                                                 : tape->data.u32_tape.offsets[tape->count];
}

/** Appends @p string at @p target, returning the byte past it; an empty one may have no start. */
static sz_ptr_t sz_py_append_(sz_ptr_t target, sz_string_view_t string) {
    if (string.length) memcpy(target, string.start, string.length);
    return target + string.length;
}

#pragma region Translation

char const doc_translate[] =                                                                              //
    "Perform transformation of a string using a look-up table.\n"                                         //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  table (str or dict): A 256-byte string, or a dict mapping characters below U+0100 to such.\n"      //
    "  inplace (bool, optional): If True, the string is modified in place, defaulting to False.\n"        //
    "\n"                                                                                                  //
    "  start (int, optional): The starting index for translation, defaulting to 0.\n"                     //
    "  end (int, optional): The ending index for translation, defaulting to the string length.\n"         //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Union[None, str, bytes]: If inplace is False, a translated copy of the [start, end) slice is\n"    //
    "    returned, otherwise None.\n"                                                                     //
    "Raises:\n"                                                                                           //
    "  ValueError: If the table is not 256 bytes long, or a dict names a character above U+00FF.\n"       //
    "  UnicodeDecodeError: If a str input translates into malformed UTF-8.\n"                             //
    "  TypeError: If the table is not a string or dictionary.\n"                                          //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('abc').translate({'a': 'A'}) == b'Abc'\n"                                               //
    "  True";

PyObject *Str_like_translate(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count < !is_member + 1 || positional_args_count > !is_member + 4) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *text_object = is_member ? self : args[0];
    PyObject *table_object = args[!is_member];
    PyObject *inplace_object = positional_args_count > !is_member + 1 ? args[!is_member + 1] : NULL;
    PyObject *start_object = positional_args_count > !is_member + 2 ? args[!is_member + 2] : NULL;
    PyObject *end_object = positional_args_count > !is_member + 3 ? args[!is_member + 3] : NULL;
    PyObject *capabilities_object = NULL;

    // Optional keyword arguments
    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "inplace") == 0 && !inplace_object) { inplace_object = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "start") == 0 && !start_object) { start_object = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0 && !end_object) { end_object = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 && !capabilities_object) {
                capabilities_object = value;
            }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) return NULL;
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    // Optional start and end arguments
    Py_ssize_t start, end;
    if (!sz_py_export_optional_index(start_object, 0, &start)) {
        PyErr_SetString(PyExc_TypeError, "start must be an integer");
        return NULL;
    }
    if (!sz_py_export_optional_index(end_object, PY_SSIZE_T_MAX, &end)) {
        PyErr_SetString(PyExc_TypeError, "end must be an integer");
        return NULL;
    }

    sz_string_view_t text;
    if (!sz_py_export_string_like(text_object, &text.start, &text.length)) {
        wrap_current_exception("First argument must be string-like");
        return NULL;
    }

    sz_align_(64) char look_up_table[256];
    if (sz_py_export_lookup_table_(table_object, look_up_table) != 0) return NULL;

    int is_inplace = inplace_object ? PyObject_IsTrue(inplace_object) : 0;
    if (is_inplace == -1) {
        PyErr_SetString(PyExc_TypeError, "The inplace argument must be a boolean");
        return NULL;
    }

    // Clamp [start, end) into the string, so negative, out-of-range or inverted windows fit
    sz_size_t normalized_offset, normalized_length;
    sz_ssize_clamp_interval(text.length, start, end, &normalized_offset, &normalized_length);
    text.start += normalized_offset;
    text.length = normalized_length;

    if (is_inplace) {
        if (sz_py_is_mutable(text_object) == sz_false_k) return NULL;
        sz_status_t const status = sz_py_lookup((sz_ptr_t)text.start, text.start, text.length, look_up_table,
                                                capabilities);
        if (status != sz_success_k) {
            sz_py_raise_status(status, "translate()");
            return NULL;
        }
        Py_RETURN_NONE;
    }

    // The table maps UTF-8 bytes, which a `str` does not store, so its result is decoded back
    if (PyUnicode_Check(text_object)) {
        sz_ptr_t translated = (sz_ptr_t)PyMem_Malloc(text.length);
        if (!translated) return PyErr_NoMemory();
        sz_status_t const status = sz_py_lookup(translated, text.start, text.length, look_up_table, capabilities);
        if (status != sz_success_k) {
            PyMem_Free(translated);
            sz_py_raise_status(status, "translate()");
            return NULL;
        }
        PyObject *translated_text = PyUnicode_DecodeUTF8(translated, (Py_ssize_t)text.length, "strict");
        PyMem_Free(translated);
        return translated_text;
    }
    PyObject *translated_bytes = PyBytes_FromStringAndSize(NULL, text.length);
    if (!translated_bytes) return NULL;
    sz_status_t const status = sz_py_lookup(PyBytes_AS_STRING(translated_bytes), text.start, text.length, look_up_table,
                                            capabilities);
    if (status != sz_success_k) {
        Py_DECREF(translated_bytes);
        sz_py_raise_status(status, "translate()");
        return NULL;
    }
    return translated_bytes;
}

char const doc_Strs_translate[] =                                                                         //
    "translate(table, *, capabilities=None) -> Strs\n"                                                    //
    "\n"                                                                                                  //
    "Map every byte of every string through a look-up table, like `bytes.translate` over each.\n"         //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  table (str or bytes or dict): A 256-byte table, or a dict mapping characters below U+0100.\n"      //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Strs: A new collection, each string as long as its source.\n"                                      //
    "Raises:\n"                                                                                           //
    "  ValueError: If the table is not 256 bytes long, or a dict names a character above U+00FF.\n"       //
    "Example:\n"                                                                                          //
    "  >>> [str(text) for text in sz.Strs(['cab', 'bee']).translate({'a': 'A', 'b': 'B'})]\n"             //
    "  ['cAB', 'Bee']";

PyObject *Strs_translate(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                         PyObject *args_names_tuple) {
    if (positional_args_count > 1)
        return PyErr_Format(PyExc_TypeError, "translate() takes at most 1 positional argument, got %zd",
                            positional_args_count);
    PyObject *table_object = positional_args_count ? args[0] : NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "table") == 0          ? &table_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                                                                                       : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "translate() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "translate() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    if (!table_object) {
        PyErr_SetString(PyExc_TypeError, "translate() missing required argument 'table'");
        return NULL;
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;
    sz_align_(64) char look_up_table[256];
    if (sz_py_export_lookup_table_(table_object, look_up_table) != 0) return NULL;

    sz_strings_view_t strings;
    sz_py_export_strings_as_view((PyObject *)self, &strings);
    sz_allocator_t allocator;
    sz_strings_span_t tape = {0};
    sz_status_t status = sz_allocator_init_heap(&allocator);
    if (status == sz_success_k)
        status = sz_strings_lookup_best(&strings, NULL, strings.count, look_up_table, &allocator, &tape, capabilities,
                                        NULL);
    return sz_py_strings_result_(status, &tape, &allocator, "translate()");
}

#pragma endregion Translation

#pragma region Escaping

/** Reads the name of an escape kind, @c json when absent, raising errors that name @p context. */
static int sz_py_export_escape_kind_(PyObject *kind_object, char const *context, sz_escape_kind_t *kind) {
    static struct {
        char const *name;
        sz_escape_kind_t kind;
    } const kinds[] = {
        {"json", sz_escape_json_k}, {"json_ascii", sz_escape_json_ascii_k},
        {"hex", sz_escape_hex_k},   {"html", sz_escape_html_k},
        {"url", sz_escape_url_k},   {"console", sz_escape_console_k},
        {"csv", sz_escape_csv_k},
    };
    *kind = sz_escape_json_k;
    if (!kind_object) return 0;
    if (!PyUnicode_Check(kind_object)) {
        PyErr_Format(PyExc_TypeError, "%s argument 'kind' must be a str", context);
        return -1;
    }
    for (sz_size_t index = 0; index != sizeof(kinds) / sizeof(kinds[0]); ++index)
        if (PyUnicode_CompareWithASCIIString(kind_object, kinds[index].name) == 0) {
            *kind = kinds[index].kind;
            return 0;
        }
    PyErr_Format(PyExc_ValueError,
                 "%s unknown kind '%U': expected 'json', 'json_ascii', 'hex', 'html', 'url', 'console' or 'csv'",
                 context, kind_object);
    return -1;
}

char const doc_Strs_escape[] =                                                                            //
    "escape(kind='json', *, capabilities=None) -> Strs\n"                                                 //
    "\n"                                                                                                  //
    "Escape every string for a text format, in one pass over the whole collection.\n"                     //
    "\n"                                                                                                  //
    "Kinds:\n"                                                                                            //
    "  json: A JSON string body, as `json.dumps(text, ensure_ascii=False)` writes it unquoted.\n"         //
    "  json_ascii: The same with every non-ASCII codepoint as `\\uXXXX`, like `json.dumps(text)`.\n"      //
    "  hex: Two lowercase digits per byte, as `bytes.hex` writes them.\n"                                 //
    "  html: `&`, `<`, `>`, `\"` and `'` as entities, like `html.escape`, with `&#39;` for `'`.\n"        //
    "  url: Percent-encoding, as `urllib.parse.quote(text, safe='')` writes it.\n"                        //
    "  console: Controls, ill-formed UTF-8 and bidirectional overrides as `\\n`, `\\xHH` or `\\uXXXX`.\n" //
    "  csv: A field quoted only around a comma, a quote, CR or LF, as `csv.writer` writes it.\n"          //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  kind (str, optional): One of the kinds above, defaulting to 'json'.\n"                             //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Strs: A new collection of the escaped strings.\n"                                                  //
    "Example:\n"                                                                                          //
    "  >>> [str(text) for text in sz.Strs(['<b>', 'Tom & Jerry']).escape('html')]\n"                      //
    "  ['&lt;b&gt;', 'Tom &amp; Jerry']";

PyObject *Strs_escape(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count, PyObject *args_names_tuple) {
    if (positional_args_count > 1)
        return PyErr_Format(PyExc_TypeError, "escape() takes at most 1 positional argument, got %zd",
                            positional_args_count);
    PyObject *kind_object = positional_args_count ? args[0] : NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "kind") == 0           ? &kind_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                                                                                       : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "escape() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "escape() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    sz_escape_kind_t kind;
    sz_capability_t capabilities;
    if (sz_py_export_escape_kind_(kind_object, "escape()", &kind) != 0) return NULL;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_strings_view_t strings;
    sz_py_export_strings_as_view((PyObject *)self, &strings);
    sz_allocator_t allocator;
    sz_strings_span_t tape = {0};
    sz_status_t status = sz_allocator_init_heap(&allocator);
    if (status == sz_success_k)
        status = sz_strings_escape_best(&strings, NULL, strings.count, kind, &allocator, &tape, capabilities, NULL);
    return sz_py_strings_result_(status, &tape, &allocator, "escape()");
}

char const doc_Strs_unescape[] =                                                                          //
    "unescape(kind='json', *, capabilities=None) -> Strs\n"                                               //
    "\n"                                                                                                  //
    "Reverse `escape` for every string, in one pass over the whole collection. The json kinds\n"          //
    "decode what `json.loads` decodes in a string body, html what `html.unescape` decodes of the\n"       //
    "five XML entities and numeric references, url what `urllib.parse.unquote_to_bytes` decodes,\n"       //
    "and csv one field as `csv.reader` reads it.\n"                                                       //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  kind (str, optional): One of the kinds `escape` takes, defaulting to 'json'.\n"                    //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Strs: A new collection of the unescaped strings.\n"                                                //
    "Raises:\n"                                                                                           //
    "  ValueError: Naming the index of the first string holding a malformed escape.\n"                    //
    "Example:\n"                                                                                          //
    "  >>> [str(text) for text in sz.Strs(['a%20b', '100%25']).unescape('url')]\n"                        //
    "  ['a b', '100%']";

PyObject *Strs_unescape(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                        PyObject *args_names_tuple) {
    if (positional_args_count > 1)
        return PyErr_Format(PyExc_TypeError, "unescape() takes at most 1 positional argument, got %zd",
                            positional_args_count);
    PyObject *kind_object = positional_args_count ? args[0] : NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "kind") == 0           ? &kind_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                                                                                       : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "unescape() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "unescape() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    sz_escape_kind_t kind;
    sz_capability_t capabilities;
    if (sz_py_export_escape_kind_(kind_object, "unescape()", &kind) != 0) return NULL;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_strings_view_t strings;
    sz_py_export_strings_as_view((PyObject *)self, &strings);
    sz_allocator_t allocator;
    sz_strings_span_t tape = {0};
    sz_size_t malformed_index = 0;
    sz_status_t status = sz_allocator_init_heap(&allocator);
    if (status == sz_success_k)
        status = sz_strings_unescape_best(&strings, NULL, strings.count, kind, &allocator, &tape, &malformed_index,
                                          capabilities, NULL);
    if (status != sz_invalid_utf8_k) return sz_py_strings_result_(status, &tape, &allocator, "unescape()");
    sz_strings_span_free(&tape, &allocator, NULL);
    PyErr_Format(PyExc_ValueError, "unescape(): the string at index %zu holds a malformed escape", malformed_index);
    return NULL;
}

#pragma endregion Escaping

#pragma region Normalization

char const doc_Strs_normalize[] =                                                                         //
    "normalize(form='NFC', *, capabilities=None) -> Strs\n"                                               //
    "\n"                                                                                                  //
    "Bring every string into a Unicode normalization form, in one pass over the whole collection,\n"      //
    "like `unicodedata.normalize` over each. 'NFKC_Casefold' is the Unicode toNFKC_Casefold, the\n"       //
    "keys search engines match on: NFKC with full case folding and no default-ignorable codepoints.\n"    //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  form (str, optional): 'NFC', 'NFD', 'NFKC', 'NFKD' or 'NFKC_Casefold', defaulting to 'NFC'.\n"     //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Strs: A new collection of the normalized strings.\n"                                               //
    "Example:\n"                                                                                          //
    "  >>> [str(text) for text in sz.Strs(['\\ufb01', 'Stra\\u00dfe']).normalize('NFKC_Casefold')]\n"     //
    "  ['fi', 'strasse']";

PyObject *Strs_normalize(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                         PyObject *args_names_tuple) {
    if (positional_args_count > 1)
        return PyErr_Format(PyExc_TypeError, "normalize() takes at most 1 positional argument, got %zd",
                            positional_args_count);
    PyObject *form_object = positional_args_count ? args[0] : NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "form") == 0           ? &form_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                                                                                       : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "normalize() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "normalize() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    sz_normal_form_t form = sz_normal_form_nfc_k;
    sz_capability_t capabilities;
    if (form_object && sz_py_export_normal_form(form_object, "normalize()", &form) != 0) return NULL;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_strings_view_t strings;
    sz_py_export_strings_as_view((PyObject *)self, &strings);
    sz_allocator_t allocator;
    sz_strings_span_t tape = {0};
    sz_status_t status = sz_allocator_init_heap(&allocator);
    if (status == sz_success_k)
        status = sz_strings_normalize_best(&strings, NULL, strings.count, form, &allocator, &tape, capabilities, NULL);
    return sz_py_strings_result_(status, &tape, &allocator, "normalize()");
}

#pragma endregion Normalization

#pragma region Dumping

char const doc_Strs_to_json[] =                                                                            //
    "to_json(ensure_ascii=True, *, capabilities=None) -> bytes\n"                                          //
    "\n"                                                                                                   //
    "Dump the strings as one compact JSON array, the bytes of\n"                                           //
    "`json.dumps(strings, ensure_ascii=ensure_ascii, separators=(',', ':')).encode()`.\n"                  //
    "\n"                                                                                                   //
    "Args:\n"                                                                                              //
    "  ensure_ascii (bool, optional): Write every non-ASCII codepoint as `\\uXXXX`, defaulting to True.\n" //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n"  //
    "Returns:\n"                                                                                           //
    "  bytes: The UTF-8 JSON text.\n"                                                                      //
    "Example:\n"                                                                                           //
    "  >>> sz.Strs(['Ada', 'Grace']).to_json()\n"                                                          //
    "  b'[\"Ada\",\"Grace\"]'";

PyObject *Strs_to_json(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                       PyObject *args_names_tuple) {
    if (positional_args_count > 1)
        return PyErr_Format(PyExc_TypeError, "to_json() takes at most 1 positional argument, got %zd",
                            positional_args_count);
    PyObject *ascii_object = positional_args_count ? args[0] : NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "ensure_ascii") == 0   ? &ascii_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                                                                                       : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "to_json() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "to_json() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    int const ensure_ascii = ascii_object ? PyObject_IsTrue(ascii_object) : 1;
    if (ensure_ascii < 0) return NULL;
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_strings_view_t strings;
    sz_py_export_strings_as_view((PyObject *)self, &strings);
    sz_allocator_t allocator;
    sz_strings_span_t tape = {0};
    sz_status_t status = sz_allocator_init_heap(&allocator);
    if (status == sz_success_k)
        status = sz_strings_escape_best(&strings, NULL, strings.count,
                                        ensure_ascii ? sz_escape_json_ascii_k : sz_escape_json_k, &allocator, &tape,
                                        capabilities, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "to_json()");
        return NULL;
    }

    // Brackets around the array, quotes around every string, and a comma between every two
    sz_size_t const count = tape.count;
    sz_size_t const bytes = 2 + count * 3 - (count != 0) + sz_py_tape_payload_(&tape);
    PyObject *result = bytes <= (sz_size_t)PY_SSIZE_T_MAX ? PyBytes_FromStringAndSize(NULL, (Py_ssize_t)bytes)
                                                          : PyErr_NoMemory();
    if (result) {
        sz_ptr_t target = PyBytes_AS_STRING(result);
        *target++ = '[';
        for (sz_size_t index = 0; index != count; ++index) {
            if (index) *target++ = ',';
            *target++ = '"';
            target = sz_py_append_(target, sz_py_tape_string_(&tape, index));
            *target++ = '"';
        }
        *target = ']';
    }
    sz_strings_span_free(&tape, &allocator, NULL);
    return result;
}

/**
 *  @brief Escapes the strings of one column, a @c Strs read in place or any other sequence of
 *      strings through a snapshot of it, into @p tape.
 *  @return 0 on success, or -1 with a Python exception set that names @p context.
 */
static int sz_py_escape_column_(PyObject *column, char const *context, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                sz_strings_span_t *tape, sz_capability_t capabilities) {
    sz_strings_view_t strings;
    sz_string_view_t *views = NULL;
    PyObject *snapshot = NULL;
    if (!sz_py_export_strings_as_view(column, &strings)) {
        // The snapshot pins every string the views point into until the escape is done
        snapshot = PySequence_Tuple(column);
        if (!snapshot) return -1;
        sz_size_t const count = (sz_size_t)PyTuple_GET_SIZE(snapshot);
        views = (sz_string_view_t *)PyMem_Malloc(count * sizeof(sz_string_view_t) + 1);
        if (!views) {
            Py_DECREF(snapshot);
            PyErr_NoMemory();
            return -1;
        }
        for (sz_size_t index = 0; index != count; ++index) {
            PyObject *const item = PyTuple_GET_ITEM(snapshot, (Py_ssize_t)index);
            if (sz_py_export_string_like(item, &views[index].start, &views[index].length)) continue;
            // A `str` that cannot be UTF-8 keeps its `UnicodeEncodeError`
            if (!PyUnicode_Check(item))
                PyErr_Format(PyExc_TypeError, "%s item %zu of a column is a %s, not a string", context, index,
                             Py_TYPE(item)->tp_name);
            PyMem_Free(views);
            Py_DECREF(snapshot);
            return -1;
        }
        sz_strings_from_string_views(views, count, &strings);
    }
    sz_status_t const status = sz_strings_escape_best(&strings, NULL, strings.count, kind, allocator, tape,
                                                      capabilities, NULL);
    PyMem_Free(views);
    Py_XDECREF(snapshot);
    if (status == sz_success_k) return 0;
    sz_py_raise_status(status, context);
    return -1;
}

/** Frees the @p count tapes of @c sz_py_escape_columns_ and the array holding them. */
static void sz_py_free_tapes_(sz_strings_span_t *tapes, Py_ssize_t count, sz_allocator_t *allocator) {
    for (Py_ssize_t index = 0; index != count; ++index) sz_strings_span_free(&tapes[index], allocator, NULL);
    PyMem_Free(tapes);
}

/**
 *  @brief Escapes the names of the mapping @p columns into the first of @p tapes, and every column
 *      into the tapes after it, checking that all columns hold equally many strings.
 *  @param[out] tapes Receives `1 + columns` tapes, freed by @c sz_py_free_tapes_.
 *  @return The number of columns, or -1 with a Python exception set and nothing left to free.
 */
static Py_ssize_t sz_py_escape_columns_(PyObject *columns, char const *context, sz_escape_kind_t kind,
                                        sz_allocator_t *allocator, sz_capability_t capabilities,
                                        sz_strings_span_t **tapes) {
    PyObject *const names = PyMapping_Keys(columns);
    if (!names) return -1;
    Py_ssize_t const columns_count = PyList_GET_SIZE(names);
    // Zeroed tapes hold no blocks, so a failure frees all of them, escaped or not
    sz_strings_span_t *const escaped = (sz_strings_span_t *)PyMem_Calloc((size_t)columns_count + 1,
                                                                         sizeof(sz_strings_span_t));
    if (!escaped) {
        Py_DECREF(names);
        PyErr_NoMemory();
        return -1;
    }
    int failed = sz_py_escape_column_(names, context, kind, allocator, &escaped[0], capabilities);
    for (Py_ssize_t column_index = 1; !failed && column_index <= columns_count; ++column_index) {
        PyObject *const name = PyList_GET_ITEM(names, column_index - 1);
        PyObject *const column = PyObject_GetItem(columns, name);
        failed = !column ||
                 sz_py_escape_column_(column, context, kind, allocator, &escaped[column_index], capabilities);
        Py_XDECREF(column);
        if (!failed && escaped[column_index].count != escaped[1].count) {
            PyErr_Format(PyExc_ValueError, "%s column %R holds %zu strings, while the first one holds %zu", context,
                         name, escaped[column_index].count, escaped[1].count);
            failed = 1;
        }
    }
    Py_DECREF(names);
    if (failed) {
        sz_py_free_tapes_(escaped, columns_count + 1, allocator);
        return -1;
    }
    *tapes = escaped;
    return columns_count;
}

char const doc_to_json[] =                                                                                 //
    "to_json(columns, ensure_ascii=True, *, capabilities=None) -> bytes\n"                                 //
    "\n"                                                                                                   //
    "Dump columns of strings as a compact JSON array of records, one object per row, the bytes of\n"       //
    "`json.dumps([dict(zip(columns, row)) for row in zip(*columns.values())], separators=(',', ':'))`.\n"  //
    "Each column is escaped in one pass, and the rows are written straight into the result.\n"             //
    "\n"                                                                                                   //
    "Args:\n"                                                                                              //
    "  columns (Mapping[str, Strs | Sequence[str]]): Columns of equal length, keyed by name.\n"            //
    "  ensure_ascii (bool, optional): Write every non-ASCII codepoint as `\\uXXXX`, defaulting to True.\n" //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n"  //
    "Returns:\n"                                                                                           //
    "  bytes: The UTF-8 JSON text.\n"                                                                      //
    "Raises:\n"                                                                                            //
    "  ValueError: If the columns differ in length.\n"                                                     //
    "Example:\n"                                                                                           //
    "  >>> sz.to_json({'name': sz.Strs(['Ada', 'Linus']), 'city': ['London', 'Helsinki']})\n"              //
    "  b'[{\"name\":\"Ada\",\"city\":\"London\"},{\"name\":\"Linus\",\"city\":\"Helsinki\"}]'";

PyObject *module_to_json(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                         PyObject *args_names_tuple) {
    sz_unused_(self);
    if (positional_args_count > 2)
        return PyErr_Format(PyExc_TypeError, "to_json() takes at most 2 positional arguments, got %zd",
                            positional_args_count);
    PyObject *columns = positional_args_count > 0 ? args[0] : NULL;
    PyObject *ascii_object = positional_args_count > 1 ? args[1] : NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "columns") == 0        ? &columns
                          : PyUnicode_CompareWithASCIIString(key, "ensure_ascii") == 0 ? &ascii_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                                                                                       : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "to_json() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "to_json() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    if (!columns) {
        PyErr_SetString(PyExc_TypeError, "to_json() missing required argument 'columns'");
        return NULL;
    }
    int const ensure_ascii = ascii_object ? PyObject_IsTrue(ascii_object) : 1;
    if (ensure_ascii < 0) return NULL;
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;
    sz_allocator_t allocator;
    sz_status_t const allocator_status = sz_allocator_init_heap(&allocator);
    if (allocator_status != sz_success_k) {
        sz_py_raise_status(allocator_status, "to_json()");
        return NULL;
    }

    sz_strings_span_t *tapes;
    Py_ssize_t const columns_count = sz_py_escape_columns_(columns, "to_json()",
                                                           ensure_ascii ? sz_escape_json_ascii_k : sz_escape_json_k,
                                                           &allocator, capabilities, &tapes);
    if (columns_count < 0) return NULL;

    // Brackets around the array, and per row the braces, the comma before the next row, every
    // `"name":"value"` cell, and the commas between cells
    sz_size_t const rows = columns_count ? tapes[1].count : 0, cells = (sz_size_t)columns_count;
    sz_size_t bytes = 2 + rows * (sz_py_tape_payload_(&tapes[0]) + cells * 6 + 2) - (rows != 0);
    for (Py_ssize_t column = 1; column <= columns_count; ++column) bytes += sz_py_tape_payload_(&tapes[column]);
    PyObject *result = bytes <= (sz_size_t)PY_SSIZE_T_MAX ? PyBytes_FromStringAndSize(NULL, (Py_ssize_t)bytes)
                                                          : PyErr_NoMemory();
    if (result) {
        sz_ptr_t target = PyBytes_AS_STRING(result);
        *target++ = '[';
        for (sz_size_t row = 0; row != rows; ++row) {
            if (row) *target++ = ',';
            *target++ = '{';
            for (sz_size_t cell = 0; cell != cells; ++cell) {
                if (cell) *target++ = ',';
                *target++ = '"';
                target = sz_py_append_(target, sz_py_tape_string_(&tapes[0], cell));
                memcpy(target, "\":\"", 3), target += 3;
                target = sz_py_append_(target, sz_py_tape_string_(&tapes[cell + 1], row));
                *target++ = '"';
            }
            *target++ = '}';
        }
        *target = ']';
    }
    sz_py_free_tapes_(tapes, columns_count + 1, &allocator);
    return result;
}

char const doc_to_csv[] =                                                                                 //
    "to_csv(columns, lineterminator='\\n', *, capabilities=None) -> bytes\n"                              //
    "\n"                                                                                                  //
    "Dump columns of strings as CSV with a header row of their names, the bytes `csv.writer` writes\n"    //
    "with its default dialect, quoting a field only around a comma, a quote, CR or LF. Each column\n"     //
    "is escaped in one pass, and the rows are written straight into the result.\n"                        //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  columns (Mapping[str, Strs | Sequence[str]]): Columns of equal length, keyed by name.\n"           //
    "  lineterminator (str, optional): '\\n' or '\\r\\n' after every row, defaulting to '\\n'.\n"         //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  bytes: The UTF-8 CSV text.\n"                                                                      //
    "Raises:\n"                                                                                           //
    "  ValueError: If the columns differ in length, or the terminator is another string.\n"               //
    "Example:\n"                                                                                          //
    "  >>> sz.to_csv({'name': ['Ada', 'Grace \"Amazing\" Hopper'], 'city': ['London', 'New York']})\n"    //
    "  b'name,city\\nAda,London\\n\"Grace \"\"Amazing\"\" Hopper\",New York\\n'";

PyObject *module_to_csv(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                        PyObject *args_names_tuple) {
    sz_unused_(self);
    if (positional_args_count > 2)
        return PyErr_Format(PyExc_TypeError, "to_csv() takes at most 2 positional arguments, got %zd",
                            positional_args_count);
    PyObject *columns = positional_args_count > 0 ? args[0] : NULL;
    PyObject *terminator_object = positional_args_count > 1 ? args[1] : NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "columns") == 0          ? &columns
                          : PyUnicode_CompareWithASCIIString(key, "lineterminator") == 0 ? &terminator_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0   ? &capabilities_object
                                                                                         : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "to_csv() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "to_csv() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    if (!columns) {
        PyErr_SetString(PyExc_TypeError, "to_csv() missing required argument 'columns'");
        return NULL;
    }
    // Only CR and LF end a row, as every field holding either is quoted
    sz_string_view_t terminator = {"\n", 1};
    if (terminator_object && PyUnicode_Check(terminator_object) &&
        PyUnicode_CompareWithASCIIString(terminator_object, "\r\n") == 0)
        terminator.start = "\r\n", terminator.length = 2;
    else if (terminator_object &&
             (!PyUnicode_Check(terminator_object) || PyUnicode_CompareWithASCIIString(terminator_object, "\n") != 0)) {
        PyErr_SetString(PyExc_ValueError, "to_csv() lineterminator must be '\\n' or '\\r\\n'");
        return NULL;
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;
    sz_allocator_t allocator;
    sz_status_t const allocator_status = sz_allocator_init_heap(&allocator);
    if (allocator_status != sz_success_k) {
        sz_py_raise_status(allocator_status, "to_csv()");
        return NULL;
    }

    sz_strings_span_t *tapes;
    Py_ssize_t const columns_count = sz_py_escape_columns_(columns, "to_csv()", sz_escape_csv_k, &allocator,
                                                           capabilities, &tapes);
    if (columns_count < 0) return NULL;

    // Every line ends with the terminator and separates its fields with commas, and as in
    // `csv.writer`, a line of one empty field writes it as `""`, not to be mistaken for no fields
    sz_size_t const rows = columns_count ? tapes[1].count : 0, cells = (sz_size_t)columns_count;
    sz_size_t const separators = cells ? cells - 1 : 0;
    sz_size_t bytes = (rows + 1) * (separators + terminator.length) + sz_py_tape_payload_(&tapes[0]);
    for (Py_ssize_t column = 1; column <= columns_count; ++column) bytes += sz_py_tape_payload_(&tapes[column]);
    for (sz_size_t line = 0; cells == 1 && line <= rows; ++line) {
        sz_string_view_t const field = line ? sz_py_tape_string_(&tapes[1], line - 1)
                                            : sz_py_tape_string_(&tapes[0], 0);
        bytes += field.length ? 0 : 2;
    }
    PyObject *result = bytes <= (sz_size_t)PY_SSIZE_T_MAX ? PyBytes_FromStringAndSize(NULL, (Py_ssize_t)bytes)
                                                          : PyErr_NoMemory();
    if (result) {
        sz_ptr_t target = PyBytes_AS_STRING(result);
        // Line zero is the header, whose fields are the names, while line `l` holds row `l - 1`
        for (sz_size_t line = 0; line <= rows; ++line) {
            for (sz_size_t cell = 0; cell != cells; ++cell) {
                sz_string_view_t const field = line ? sz_py_tape_string_(&tapes[cell + 1], line - 1)
                                                    : sz_py_tape_string_(&tapes[0], cell);
                if (cell) *target++ = ',';
                if (cells == 1 && !field.length) memcpy(target, "\"\"", 2), target += 2;
                target = sz_py_append_(target, field);
            }
            target = sz_py_append_(target, terminator);
        }
    }
    sz_py_free_tapes_(tapes, columns_count + 1, &allocator);
    return result;
}

#pragma endregion Dumping
