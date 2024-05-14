import test from 'node:test';
import bindings from 'bindings';
import assert from 'node:assert';

const stringzilla = bindings('../../build/Release/stringzilla');

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

test('Str Count - Empty String Inputs', () => {
    const a = new stringzilla.Str('hello world');
    const b = new stringzilla.Str('hi');
    const empty = new stringzilla.Str('');

    assert.strictEqual(a.count(empty), 0n);
    assert.strictEqual(empty.count(b), 0n);
    assert.strictEqual(empty.count(empty), 0n);
});
test('Str.count - No Occurence', () => {
    const a = new stringzilla.Str('hello world');
    const b = new stringzilla.Str('hi');

    const result_1 = a.count(b);
    assert.strictEqual(result_1, 0n);
});
test('Str.count - Multiple Occurrences with Overlap Test', () => {
    const a = new stringzilla.Str('abababab');
    const b = new stringzilla.Str('aba');

    const result_2 = a.count(b, true);
    assert.strictEqual(result_2, 3n);
    const result_1 = a.count(b);
    assert.strictEqual(result_1, 2n);

});

test('Str.count - Multiple Occurrences', () => {
    const a = new stringzilla.Str('abigababzzzzzzzzzzzzzzzzzzzbigzzzzzzzzzzzzfdsafbig');
    const b = new stringzilla.Str('big');

    const res = a.count(b, true);
    assert.strictEqual(res, 3n);

});
test('Str.count - Single Occurrence', () => {
    const a = new stringzilla.Str('azigababzzzzzzzzzzzzzzzzzzzzigzzzzzzzzzzzzfdsafbig');
    const b = new stringzilla.Str('big');

    const res = a.count(b, true);
    assert.strictEqual(res, 1n);

});


test('Str.find - Positive Case', () => {
    const a = new stringzilla.Str('Can you ifnd me here with find');
    const b = new stringzilla.Str('find');
    assert.strictEqual(a.find(b), 26n);
});

test('Str.find - Negative Case (Word Not Found)', () => {
    const a = new stringzilla.Str('Can you ifnd me here with find');
    const b = new stringzilla.Str('z');
    assert.strictEqual(a.find(b), -1n);
});

test('Str.find - Negative Case (Empty String Inputs)', () => {
    const a = new stringzilla.Str('hello world');
    const b = new stringzilla.Str('hi');
    const empty = new stringzilla.Str('');

    assert.strictEqual(a.find(empty),     0n);
    assert.strictEqual(empty.find(b),    -1n);
    assert.strictEqual(empty.find(empty), 0n);
});
test('Str.rfind', () => {
    const a = new stringzilla.Str('Can you ifnd me here with find');
    const b = new stringzilla.Str('n');
    const can = new stringzilla.Str('Can');
    const z = new stringzilla.Str('z');
    const empty = new stringzilla.Str('');
    assert.strictEqual(a.rfind(b), 28n);
    assert.strictEqual(a.rfind(z), -1n);
    assert.strictEqual(a.rfind(can), 0n);
    assert.strictEqual(a.rfind(empty), 0n);
    assert.strictEqual(empty.rfind(z), -1n);
    assert.strictEqual(empty.rfind(empty), 0n);
});
test('Str.startswith', () => {
    const a = new stringzilla.Str('Can you ifnd me here with find');
    const b = new stringzilla.Str('n');
    const can = new stringzilla.Str('Can');
    const empty = new stringzilla.Str('');
    assert.strictEqual(a.startswith(b), false);
    assert.strictEqual(a.startswith(can), true);
    assert.strictEqual(a.startswith(empty), true);
    assert.strictEqual(empty.startswith(a), false);
    assert.strictEqual(empty.startswith(empty), true);
});
test('Str.endswith', () => {
    const a = new stringzilla.Str('Can you ifnd me here with find');
    const b = new stringzilla.Str('n');
    const can = new stringzilla.Str('find');
    const empty = new stringzilla.Str('');
    assert.strictEqual(a.endswith(b), false);
    assert.strictEqual(a.endswith(can), true);
    assert.strictEqual(a.endswith(empty), true);
    assert.strictEqual(empty.endswith(a), false);
    assert.strictEqual(empty.endswith(empty), true);
});


