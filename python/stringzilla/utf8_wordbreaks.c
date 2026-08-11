/**
 *  @brief UAX-29 word segmentation of UTF-8 text.
 *  @file python/stringzilla/utf8_wordbreaks.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

char const doc_utf8_wordbreaks[] =                                                     //
    "utf8_wordbreaks(string, /)\n"                                                     //
    "\n"                                                                               //
    "Return an iterator yielding words per Unicode UAX-29 word boundary rules.\n"      //
    "Unlike str.split(), this is UAX-29 compliant and supports all Unicode scripts.\n" //
    "\n"                                                                               //
    "Args:\n"                                                                          //
    "    string: The input UTF-8 string to split into words.\n"                        //
    "\n"                                                                               //
    "Returns:\n"                                                                       //
    "    Iterator yielding Str objects for each word.\n\n"                             //
    "\n"                                                                               //
    "Example:\n"                                                                       //
    "  >>> # Stream UAX-29 word tokens lazily:\n"                                      //
    "  >>> 'world' in (str(w) for w in sz.utf8_wordbreaks('Hi, world'))\n"             //
    "  True";

PyObject *Str_like_utf8_wordbreaks(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *kwnames) {
    int min_args = 1, max_args = 1;
    if (positional_args_count < min_args || positional_args_count > max_args) {
        PyErr_Format(PyExc_TypeError, "utf8_wordbreaks() requires %zd to %zd arguments", min_args, max_args);
        return NULL;
    }

    sz_unused_(self);
    return Utf8Boundaries_make_(&Utf8WordbreaksType, args[0], sz_utf8_wordbreaks);
}

static char const doc_Utf8Wordbreaks[] =                                  //
    "Utf8Wordbreaks(string, ...)\n"                                       //
    "\n"                                                                  //
    "UTF-8 aware word boundary iterator per Unicode UAX-29 algorithm.\n"  //
    "Yields words (text segments between consecutive word boundaries).\n" //
    "\n"                                                                  //
    "Created by:\n"                                                       //
    "  - Str.utf8_wordbreaks()\n"                                         //
    "  - sz.utf8_wordbreaks()\n"                                          //
    "\n"                                                                  //
    "UAX-29 Word_Break rules implemented:\n"                              //
    "  - WB3: CR x LF (no break)\n"                                       //
    "  - WB4: Ignore Extend/Format/ZWJ\n"                                 //
    "  - WB5-WB13: Letter, number, punctuation rules\n"                   //
    "  - WB15-WB16: Regional Indicator pairs\n\n"                         //
    "\n"                                                                  //
    "Example:\n"                                                          //
    "  >>> len(list(sz.utf8_wordbreaks('Hi there'))) >= 2\n"              //
    "  True";

PyTypeObject Utf8WordbreaksType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Utf8Wordbreaks",
    .tp_basicsize = sizeof(Utf8Boundaries),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Utf8Boundaries_dealloc_,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = doc_Utf8Wordbreaks,
    .tp_iter = Utf8Boundaries_iter_,
    .tp_iternext = (iternextfunc)Utf8Boundaries_next_,
};
