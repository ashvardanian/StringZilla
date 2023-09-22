import test from 'node:test';
import bindings from 'bindings';
import assert from 'node:assert';

const stringzilla = bindings('stringzilla');

test('Find Word in Text - Positive Case', () => {
    const result = stringzilla.find('hello world, hello john', 'world');

    assert.strictEqual(result, 6n);
});

test('Find Word in Text - Negative Case (Word Not Found)', () => {
    const result = stringzilla.find('hello world', 'hi');

    assert.strictEqual(result, -1n);
});

test('Find Word in Text - Negative Case (Empty String Inputs)', () => {
    const result_1 = stringzilla.find('hello world', '');
    assert.strictEqual(result_1, -1n);

    const result_2 = stringzilla.find('', 'a');
    assert.strictEqual(result_2, -1n);

    const result_3 = stringzilla.find('', '');
    assert.strictEqual(result_2, -1n);
});
