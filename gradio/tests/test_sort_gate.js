const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(require('node:path').join(__dirname, '../ui/assets/editor.js'), 'utf8');
const gate = source.slice(source.indexOf('const canEditReadingOrder'), source.indexOf('const orderedClearTargets'));
const context = {props: {value: {step: 3, contentVerified: true, characterCount: 2}}, localBoxes: {a: {}, b: {}, c: {}}, mismatchConfirmationInvalidated: false};
vm.createContext(context);
vm.runInContext(gate, context);
const allowed = () => vm.runInContext('canSortBoxes()', context);
assert.equal(allowed(), false);
Object.assign(context.props.value, {mismatchConfirmed: true, mismatchBoxCount: 3});
assert.equal(allowed(), true);
context.mismatchConfirmationInvalidated = true;
assert.equal(allowed(), false);
context.mismatchConfirmationInvalidated = false;
delete context.localBoxes.c;
assert.equal(allowed(), true); // Counts match; confirmation is unnecessary.
context.localBoxes.d = {}; context.localBoxes.e = {};
assert.equal(allowed(), false); // Previous confirmation covers a different count.
context.props.value.contentVerified = false;
assert.equal(allowed(), false);
context.props.value.contentVerified = true; context.props.value.step = 4;
assert.equal(allowed(), false);
console.log('Sort gate checks passed');
