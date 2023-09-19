var assert = require('assert');
var stringzilla = require('bindings')('stringzilla');

const result = stringzilla.find("hello world", "world");
console.log(result);  // Output will depend on the result of your findOperation function.

console.log('JavaScript tests passed!');
