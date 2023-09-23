import test from 'node:test';
import bindings from 'bindings';
import assert from 'node:assert';

const stringzilla = bindings('stringzilla');

test('Count Words - Single Occurrence', () => {
    const result = stringzilla.countSubstr('hello world', 'world');

    assert.strictEqual(result, 1n);
});

test('Count Words - Multiple Occurrence', () => {
    const result = stringzilla.countSubstr('hello world, hello John', 'hello');

    assert.strictEqual(result, 2n);
});

test('Count Words - Multiple Occurrences with Overlap Test', () => {
    const result_1 = stringzilla.countSubstr('abababab', 'aba');

    assert.strictEqual(result_1, 2n);

    const result_2 = stringzilla.countSubstr('abababab', 'aba', true);

    assert.strictEqual(result_2, 3n);
});

test('Count Words - No Occurrence', () => {
    const result = stringzilla.countSubstr('hello world', 'hi');

    assert.strictEqual(result, 0n);
});

test('Count Words - Empty String Inputs', () => {
    const result_1 = stringzilla.countSubstr('hello world', '');
    assert.strictEqual(result_1, 0n);

    const result_2 = stringzilla.countSubstr('', 'hi');
    assert.strictEqual(result_2, 0n);

    const result_3 = stringzilla.countSubstr('', '');
    assert.strictEqual(result_3, 0n);
});
