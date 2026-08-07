/**
 *  @brief UAX-29 grapheme cluster segmentation of UTF-8 text.
 *  @file python/stringzilla/utf8_graphemes.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

char const doc_utf8_graphemes[] =                                                    //
    "utf8_graphemes(string, /)\n"                                                    //
    "\n"                                                                             //
    "Return an iterator yielding grapheme clusters per Unicode UAX-29 rules.\n"      //
    "A grapheme cluster is a user-perceived character (e.g. a base plus combining\n" //
    "marks, or an emoji ZWJ sequence) and may span several code points.\n"           //
    "\n"                                                                             //
    "Args:\n"                                                                        //
    "    string: The input UTF-8 string to split into grapheme clusters.\n"          //
    "\n"                                                                             //
    "Returns:\n"                                                                     //
    "    Iterator yielding Str objects for each grapheme cluster.\n\n"               //
    "\n"                                                                             //
    "Example:\n"                                                                     //
    "  >>> # Stream UAX-29 grapheme clusters lazily:\n"                              //
    "  >>> [str(g) for g in sz.utf8_graphemes('abc')]\n"                             //
    "  ['a', 'b', 'c']";

PyObject *Str_like_utf8_graphemes(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *kwnames) {
    int min_args = 1, max_args = 1;
    if (positional_args_count < min_args || positional_args_count > max_args) {
        PyErr_Format(PyExc_TypeError, "utf8_graphemes() requires %zd to %zd arguments", min_args, max_args);
        return NULL;
    }

    sz_unused_(self);
    return Utf8Boundaries_make_(&Utf8GraphemesType, args[0], sz_utf8_graphemes);
}

static char const doc_Utf8Graphemes[] =                                              //
    "Utf8Graphemes(string, ...)\n"                                                   //
    "\n"                                                                             //
    "UTF-8 aware grapheme cluster boundary iterator per Unicode UAX-29 algorithm.\n" //
    "Yields grapheme clusters (user-perceived characters).\n"                        //
    "\n"                                                                             //
    "Created by:\n"                                                                  //
    "  - Str.utf8_graphemes()\n"                                                     //
    "  - sz.utf8_graphemes()\n"                                                      //
    "\n"                                                                             //
    "UAX-29 Grapheme_Cluster_Break rules implemented:\n"                             //
    "  - GB3: CR x LF (no break)\n"                                                  //
    "  - GB4-GB5: Control/CR/LF breaks\n"                                            //
    "  - GB6-GB8: Hangul syllable sequences\n"                                       //
    "  - GB9-GB9c: Extend/ZWJ/SpacingMark, Indic conjuncts\n"                        //
    "  - GB11-GB12: Emoji ZWJ and Regional Indicator pairs\n\n"                      //
    "\n"                                                                             //
    "Example:\n"                                                                     //
    "  >>> len(list(sz.utf8_graphemes('abc'))) == 3\n"                               //
    "  True";

PyTypeObject Utf8GraphemesType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Utf8Graphemes",
    .tp_basicsize = sizeof(Utf8Boundaries),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Utf8Boundaries_dealloc_,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = doc_Utf8Graphemes,
    .tp_iter = Utf8Boundaries_iter_,
    .tp_iternext = (iternextfunc)Utf8Boundaries_next_,
};
