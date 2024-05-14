const compiled = require('bindings')('stringzilla');

module.exports = {
    /**
     * Searches for a short string in a long one.
     * 
     * @param {string} haystack 
     * @param {string} needle 
     * @returns {bigint}
     */
    indexOf: compiled.indexOf,
    indexOfB: compiled.indexOfB,

    /**
     * Searches for a substring in a larger string.
     * 
     * @param {string} haystack 
     * @param {string} needle 
     * @param {boolean} overlap 
     * @returns {bigint}
     */
    count: compiled.count,

    Str: compiled.Str
};

