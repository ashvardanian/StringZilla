import test from 'node:test';
import bindings from 'bindings';
import assert from 'node:assert';

const stringzilla = bindings('stringzilla');

test('Find Word in Text - Positive Case', () => {
    const result = stringzilla.indexOf('hello world, hello john', 'hello');

    assert.strictEqual(result, 0n);
});

test('Find Word in Text - Negative Case (Word Not Found)', () => {
    const result_1 = stringzilla.indexOf('ha', 'aaa');
    assert.strictEqual(result_1, -1n);

    const result_2 = stringzilla.indexOf('g', 'a');
    assert.strictEqual(result_2, -1n);
});

test('Find Word in Text - Negative Case (Empty String Inputs)', () => {
    const result_1 = stringzilla.indexOf('hello world', '');
    assert.strictEqual(result_1, 0n);

    const result_2 = stringzilla.indexOf('', 'a');
    assert.strictEqual(result_2, -1n);

    const result_3 = stringzilla.indexOf('', '');
    assert.strictEqual(result_3, 0n);
});


test('Count Words - Single Occurrence', () => {
    const result = stringzilla.count('hello world', 'world');

    assert.strictEqual(result, 1n);
});

test('Count Words - Multiple Occurrence', () => {
    const result = stringzilla.count('hello world, hello John', 'hello');

    assert.strictEqual(result, 2n);
});

test('Count Words - Multiple Occurrences with Overlap Test', () => {
    const result_1 = stringzilla.count('abababab', 'aba');

    assert.strictEqual(result_1, 2n);

    const result_2 = stringzilla.count('abababab', 'aba', true);

    assert.strictEqual(result_2, 3n);
});

test('Count Words - No Occurrence', () => {
    const result = stringzilla.count('hello world', 'hi');

    assert.strictEqual(result, 0n);
});

test('Count Words - Empty String Inputs', () => {
    const result_1 = stringzilla.count('hello world', '');
    assert.strictEqual(result_1, 0n);

    const result_2 = stringzilla.count('', 'hi');
    assert.strictEqual(result_2, 0n);

    const result_3 = stringzilla.count('', '');
    assert.strictEqual(result_3, 0n);
});
