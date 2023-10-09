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
