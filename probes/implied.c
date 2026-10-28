/**
 *  @file probes/implied.c
 *  @author Ash Vardanian
 *  @date October 28, 2026
 *  @brief Whether the floor's predefined macros imply the macro @c STRINGZILLA_PROBE_MACRO names.
 */
#include "stringzilla/types.h"
#if !STRINGZILLA_PROBE_MACRO
#error
#endif
