#include <stringzillas/stringzillas.h>

#include <stddef.h>
#include <stdlib.h>

static sz_size_t allocated_bytes = 0;
static sz_size_t freed_bytes = 0;

static void *counting_allocate(sz_size_t bytes, void *handle) {
    (void)handle;
    allocated_bytes += bytes;
    return malloc(bytes ? bytes : 1);
}

static void counting_free(void *address, sz_size_t bytes, void *handle) {
    (void)handle;
    freed_bytes += bytes;
    free(address);
}

int main(void) {
    char const *error_message = NULL;
    szs_device_scope_t device = NULL;
    if (szs_device_scope_init_cpu_cores(1, &device, &error_message) != sz_success_k) return 1;

    char const dictionary_data[] = "bookbackbookboon";
    sz_u32_t const dictionary_offsets[] = {0, 4, 8, 12, 16};
    sz_sequence_u32tape_t dictionary = {dictionary_data, dictionary_offsets, 4};
    szs_levenshtein_index_t index = NULL;
    sz_memory_allocator_t allocator = {counting_allocate, counting_free, NULL};
    if (szs_levenshtein_index_init_u32tape(&dictionary, 2, &allocator, sz_caps_sp_k, &index, &error_message) !=
        sz_success_k)
        return 2;

    char const queries_data[] = "cookbook";
    sz_u32_t const queries_offsets[] = {0, 4, 8};
    sz_sequence_u32tape_t queries = {queries_data, queries_offsets, 2};
    sz_u64_t query_ids[4];
    sz_u32_t dictionary_ids[4];
    sz_u8_t distances[4];
    sz_size_t matches_count = 0;
    if (szs_levenshtein_index_find_u32tape(index, device, &queries, 1, query_ids, dictionary_ids, distances, 4,
                                           &matches_count, &error_message) != sz_unexpected_dimensions_k)
        return 3;
    if (matches_count != 5) return 4;

    matches_count = 0;
    if (szs_levenshtein_index_find_u32tape(index, device, &queries, 1, NULL, NULL, NULL, 0, &matches_count,
                                           &error_message) != sz_unexpected_dimensions_k)
        return 5;
    if (matches_count != 5) return 6;

    sz_u64_t query_ids_full[5];
    sz_u32_t dictionary_ids_full[5];
    sz_u8_t distances_full[5];
    if (szs_levenshtein_index_find_u32tape(index, device, &queries, 1, query_ids_full, dictionary_ids_full,
                                           distances_full, 5, &matches_count, &error_message) != sz_success_k)
        return 7;
    sz_size_t cook_matches = 0, book_matches = 0;
    for (sz_size_t match = 0; match != matches_count; ++match) {
        if (distances_full[match] > 1 || dictionary_ids_full[match] >= 4 || query_ids_full[match] >= 2) return 8;
        if (query_ids_full[match] == 0) ++cook_matches;
        else ++book_matches;
    }
    if (cook_matches != 2 || book_matches != 3) return 9;
    szs_levenshtein_index_free(index);
    if (!allocated_bytes || allocated_bytes != freed_bytes) return 10;

    char const utf8_dictionary_data[] = "caf\xC3\xA9" "cafe" "\xE5\x92\x96\xE5\x95\xA1"
                                        "\xE5\x92\x96\xE9\x9D\x9E";
    sz_u32_t const utf8_dictionary_offsets[] = {0, 5, 9, 15, 21};
    sz_sequence_u32tape_t utf8_dictionary = {utf8_dictionary_data, utf8_dictionary_offsets, 4};
    szs_levenshtein_index_utf8_t utf8_index = NULL;
    if (szs_levenshtein_index_utf8_init_u32tape(&utf8_dictionary, 2, NULL, sz_caps_sp_k, &utf8_index,
                                                &error_message) != sz_success_k)
        return 11;
    char const utf8_queries_data[] = "\xE5\x92\x96\xE5\x95\xA1" "\xF0\x9F";
    sz_u32_t const utf8_queries_offsets[] = {0, 6, 8};
    sz_sequence_u32tape_t utf8_queries = {utf8_queries_data, utf8_queries_offsets, 2};
    matches_count = 0;
    if (szs_levenshtein_index_utf8_find_u32tape(utf8_index, device, &utf8_queries, 1, query_ids_full,
                                                dictionary_ids_full, distances_full, 5, &matches_count,
                                                &error_message) != sz_invalid_utf8_k)
        return 12;
    if (matches_count != 0) return 13;
    szs_levenshtein_index_utf8_free(utf8_index);
    szs_device_scope_free(device);
    return 0;
}
