/**
 *  @brief UAX-29 sentence segmentation of UTF-8 text.
 *  @file python/stringzilla/utf8_sentences.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

char const doc_utf8_sentences[] =                                                         //
    "utf8_sentences(string, /)\n"                                                         //
    "\n"                                                                                  //
    "Return an iterator yielding sentences per Unicode UAX-29 sentence boundary rules.\n" //
    "UAX-29 compliant and Unicode-script aware, unlike naive period splitting.\n"         //
    "\n"                                                                                  //
    "Args:\n"                                                                             //
    "    string: The input UTF-8 string to split into sentences.\n"                       //
    "\n"                                                                                  //
    "Returns:\n"                                                                          //
    "    Iterator yielding Str objects for each sentence.\n\n"                            //
    "\n"                                                                                  //
    "Example:\n"                                                                          //
    "  >>> # Stream UAX-29 sentences lazily:\n"                                           //
    "  >>> len(list(sz.utf8_sentences('Hi. Bye.'))) >= 2\n"                               //
    "  True";

PyObject *Str_like_utf8_sentences(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *kwnames) {
    int min_args = 1, max_args = 1;
    if (positional_args_count < min_args || positional_args_count > max_args) {
        PyErr_Format(PyExc_TypeError, "utf8_sentences() requires %zd to %zd arguments", min_args, max_args);
        return NULL;
    }

    sz_unused_(self);
    return Utf8Boundaries_make_(&Utf8SentencesType, args[0], sz_utf8_sentences);
}

static char const doc_Utf8Sentences[] =                                           //
    "Utf8Sentences(string, ...)\n"                                                //
    "\n"                                                                          //
    "UTF-8 aware sentence boundary iterator per Unicode UAX-29 algorithm.\n"      //
    "Yields sentences (text segments between consecutive sentence boundaries).\n" //
    "\n"                                                                          //
    "Created by:\n"                                                               //
    "  - Str.utf8_sentences()\n"                                                  //
    "  - sz.utf8_sentences()\n"                                                   //
    "\n"                                                                          //
    "UAX-29 Sentence_Break rules implemented:\n"                                  //
    "  - SB3: CR x LF (no break)\n"                                               //
    "  - SB4: ParaSep breaks\n"                                                   //
    "  - SB6-SB8: ATerm/STerm sentence-final sequences\n"                         //
    "  - SB9-SB11: Close/Sp/ParaSep continuations\n"                              //
    "  - SB998: Otherwise no break\n\n"                                           //
    "\n"                                                                          //
    "Example:\n"                                                                  //
    "  >>> len(list(sz.utf8_sentences('Hi. Bye.'))) >= 2\n"                       //
    "  True";

PyTypeObject Utf8SentencesType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Utf8Sentences",
    .tp_basicsize = sizeof(Utf8Boundaries),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Utf8Boundaries_dealloc_,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = doc_Utf8Sentences,
    .tp_iter = Utf8Boundaries_iter_,
    .tp_iternext = (iternextfunc)Utf8Boundaries_next_,
};
