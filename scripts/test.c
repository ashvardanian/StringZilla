#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <stringzilla/stringzilla.h>

#define MAX_LENGTH 300
#define MIN_LENGTH 3
#define ASCII_LOWERCASE "abcdefghijklmnopqrstuvwxyz"
#define VARIABILITY 25

// Utility function to populate random string in a buffer
void populate_random_string(char *buffer, int length, int variability) {
    for (int i = 0; i < length; i++) { buffer[i] = ASCII_LOWERCASE[rand() % variability]; }
    buffer[length] = '\0';
}

// Test function for sz_find_substring
void test_sz_find_substring() {
    char buffer[MAX_LENGTH + 1];
    char pattern[6]; // Maximum length of 5 + 1 for '\0'

    for (int length = MIN_LENGTH; length < MAX_LENGTH; length++) {
        for (int variability = 1; variability < VARIABILITY; variability++) {
            populate_random_string(buffer, length, variability);

            sz_string_view_t haystack;
            haystack.start = buffer;
            haystack.length = length;

            int pattern_length = rand() % 5 + 1;
            populate_random_string(pattern, pattern_length, variability);

            sz_string_view_t needle;
            needle.start = pattern;
            needle.length = pattern_length;

            // Comparing the result of your function with the standard library function.
            sz_string_start_t result_libc = strstr(buffer, pattern);
            sz_string_start_t result_stringzilla =
                sz_find_substring(haystack.start, haystack.length, needle.start, needle.length);

            assert(((result_libc == NULL) ^ (result_stringzilla == NULL)) && "Test failed for sz_find_substring");
        }
    }
}

int main() {
    srand((unsigned int)time(NULL));

    test_sz_find_substring();
    // Add calls to other test functions as you implement them

    printf("All tests passed!\n");
    return 0;
}
