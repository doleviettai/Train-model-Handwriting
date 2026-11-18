// Regression checks for accepting a result after adding a diacritic or pressing Space twice.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/seq/serve_seq.py', 'utf8');
const html = source.match(/HTML = r"""([\s\S]*?)"""/)[1];
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const context2d = new Proxy({}, { get: () => () => {} });
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    listeners: {}, value: '', checked: true, style: {}, textContent: '', innerHTML: '', disabled: false,
    addEventListener(name, fn) { this.listeners[name] = fn; },
    getContext() { return context2d; },
    getBoundingClientRect() { return {left: 0, top: 0, width: 900, height: 360}; },
    setPointerCapture() {},
  });
  return elements.get(id);
}
const sandbox = {
  document: {getElementById: element, addEventListener() {}},
  window: {devicePixelRatio: 1, addEventListener() {}},
  setTimeout() {return 1;}, clearTimeout() {}, console,
};
vm.createContext(sandbox);
vm.runInContext(script, sandbox);
const run = code => vm.runInContext(code, sandbox);
const response = text => ({ok: true, json: async () => ({text, chars: [{c: text, p: 0.8}], confidence: 0.8, candidates: []})});

(async () => {
  let resolvePreview;
  sandbox.fetch = () => new Promise(resolve => {resolvePreview = resolve;});
  run('strokes = [[[0,0],[1,1]]]; strokeVersion = 1;');
  const pending = run('preview()');
  element('pad').listeners.pointerdown({clientX: 2, clientY: 2, pointerId: 1});
  resolvePreview(response('o'));
  await pending;
  assert.equal(run('lastResult'), null, 'a stale o result survived adding another stroke');
  assert.equal(element('live').textContent, '...');

  let requests = 0;
  sandbox.fetch = async (url, options) => {
    requests++;
    const payload = JSON.parse(options.body);
    assert.equal(payload.expected, undefined, 'expected answer leaked into recognition');
    return response('ô');
  };
  run("current = null; lastResult = {text:'o',chars:[],version:strokeVersion-1};");
  await Promise.all([run('commitWord()'), run('commitWord()')]);
  assert.equal(requests, 1);
  assert.equal(run('words.length'), 1, 'double Space committed twice');
  assert.equal(run('words[0].text'), 'ô', 'commit reused the result before a diacritic was added');

  let resolveCommit;
  sandbox.fetch = () => new Promise(resolve => {resolveCommit = resolve;});
  run('strokes = [[[0,0],[1,1]]]; strokeVersion++;');
  const committing = run('commitWord()');
  run('clearCurrent()');
  resolveCommit(response('a'));
  await committing;
  assert.equal(run('words.length'), 1, 'clearing did not cancel the pending commit');
  assert.equal(element('btnNext').disabled, false);
  console.log('UI regression checks passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
