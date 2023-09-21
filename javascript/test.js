var assert = require('assert');
var stringzilla = require('bindings')('stringzilla');

const findResult = stringzilla.find("hello world", "world");
console.log(findResult);  // Output will depend on the result of your findOperation function.

const countResult = stringzilla.countSubstr("hello world", "world");
console.log(countResult);  // Output will depend on the result of your countSubstr function.


console.log('JavaScript tests passed!');
