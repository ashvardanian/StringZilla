/**
 *  @brief UAX-14 line break opportunities in UTF-8 text.
 *  @file python/stringzilla/utf8_linebreaks.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

char const doc_utf8_linebreaks[] =                                                   //
    "utf8_linebreaks(string, /)\n"                                                   //
    "\n"                                                                             //
    "Return an iterator yielding segments at line-break opportunities per UAX-14.\n" //
    "Each segment ends at a line-break opportunity (a soft wrap point).\n"           //
    "For hard-line splitting (str.splitlines()), use utf8_split_newlines().\n"       //
    "\n"                                                                             //
    "Args:\n"                                                                        //
    "    string: The input UTF-8 string to split at line-break opportunities.\n"     //
    "\n"                                                                             //
    "Returns:\n"                                                                     //
    "    Iterator yielding Str objects for each line-break-opportunity segment.\n\n" //
    "\n"                                                                             //
    "Example:\n"                                                                     //
    "  >>> # Stream UAX-14 line-break opportunities lazily:\n"                       //
    "  >>> len(list(sz.utf8_linebreaks('a\\nb'))) >= 2\n"                            //
    "  True";

PyObject *Str_like_utf8_linebreaks(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *kwnames) {
    int min_args = 1, max_args = 1;
    if (positional_args_count < min_args || positional_args_count > max_args) {
        PyErr_Format(PyExc_TypeError, "utf8_linebreaks() requires %zd to %zd arguments", min_args, max_args);
        return NULL;
    }

    sz_unused_(self);
    return Utf8Boundaries_make_(&Utf8LinebreaksType, args[0], sz_utf8_linebreaks);
}

static char const doc_Utf8Linebreaks[] =                                          //
    "Utf8Linebreaks(string, ...)\n"                                               //
    "\n"                                                                          //
    "UTF-8 aware line-break-opportunity iterator per Unicode UAX-14 algorithm.\n" //
    "Yields Str views for each line-break-opportunity segment (a soft wrap).\n"   //
    "For hard-line splitting (str.splitlines()), use utf8_split_newlines().\n"    //
    "\n"                                                                          //
    "Created by:\n"                                                               //
    "  - Str.utf8_linebreaks()\n"                                                 //
    "  - sz.utf8_linebreaks()\n"                                                  //
    "\n"                                                                          //
    "UAX-14 Line_Break rules implemented:\n"                                      //
    "  - LB4-LB6: Mandatory breaks (BK, CR, LF, NL)\n"                            //
    "  - LB7-LB8: Spaces and ZWSP break opportunities\n"                          //
    "  - LB9-LB14: Combining marks, opening/closing punctuation\n"                //
    "  - LB15-LB31: Quotation, numbers, words, CJK rules\n\n"                     //
    "\n"                                                                          //
    "Example:\n"                                                                  //
    "  >>> len(list(sz.utf8_linebreaks('a\\nb'))) >= 2\n"                         //
    "  True";

PyTypeObject Utf8LinebreaksType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Utf8Linebreaks",
    .tp_basicsize = sizeof(Utf8Boundaries),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Utf8Boundaries_dealloc_,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = doc_Utf8Linebreaks,
    .tp_iter = Utf8Boundaries_iter_,
    .tp_iternext = (iternextfunc)Utf8Boundaries_next_,
};
