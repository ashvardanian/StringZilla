
/**
 * Searches for a short string in a long one.
 * 
 * @param {string} haystack 
 * @param {string} needle 
 */
export function find(haystack: string, needle: string): bigint;

/**
 * Searches for a substring in a larger string.
 * 
 * @param {string} haystack 
 * @param {string} needle 
 * @param {boolean} overlap 
 */
export function count(haystack: string, needle: string, overlap: boolean): bigint;
