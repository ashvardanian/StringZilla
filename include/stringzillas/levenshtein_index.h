/**
 *  @brief Shared C layout for exact sparse Levenshtein dictionary matches.
 *  @file include/stringzillas/levenshtein_index.h
 *  @author Guillaume de Rouville
 */
#ifndef STRINGZILLAS_LEVENSHTEIN_INDEX_H_
#define STRINGZILLAS_LEVENSHTEIN_INDEX_H_

#include <stringzilla/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct szs_levenshtein_index_match_t {
    sz_u32_t id;
    sz_u8_t distance;
    sz_u8_t reserved[3];
} szs_levenshtein_index_match_t;

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLAS_LEVENSHTEIN_INDEX_H_
