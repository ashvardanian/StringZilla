const compiled = require('bindings')('stringzilla');

module.exports = {
    /**
     * Searches for a short string in a long one.
     * 
     * @param {string} haystack 
     * @param {string} needle 
     * @returns {bigint}
     */
    find: compiled.find,

    /**
     * Searches for a substring in a larger string.
     * 
     * @param {string} haystack 
     * @param {string} needle 
     * @param {boolean} overlap 
     * @returns {bigint}
     */
    count: compiled.count
};
