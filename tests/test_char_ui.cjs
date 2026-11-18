const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/serve.py', 'utf8');
const html = source.match(/HTML = r"""([\s\S]*?)"""/)[1];
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const context2d = new Proxy({}, {get: () => () => {}});
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    listeners: {}, value: '', checked: false, style: {}, textContent: '', innerHTML: '',
    addEventListener(name, fn) {this.listeners[name] = fn;},
    getContext() {return context2d;},
    getBoundingClientRect() {return {left: 0, top: 0, width: 500, height: 500};},
    setPointerCapture() {},
  });
  return elements.get(id);
}
const sandbox = {document: {getElementById: element}, window: {devicePixelRatio: 1, addEventListener() {}},
  setTimeout() {return 1;}, clearTimeout() {}, console};
vm.createContext(sandbox);
vm.runInContext(script, sandbox);
const run = code => vm.runInContext(code, sandbox);
const response = text => ({ok: true, json: async () => ({best: text, raw_char: text, confidence: 0.7,
  candidates: [{char: text, prob: 0.7}], n_strokes: 1, base: [], modifier: [], tone: [], uncertain: false})});

(async () => {
  let resolve;
  sandbox.fetch = () => new Promise(r => {resolve = r;});
  run('strokes = [[[0,0],[1,1]]];');
  const pending = run('recognize()');
  element('pad').listeners.pointerdown({clientX: 200, clientY: 50, pointerId: 1});
  resolve(response('a'));
  await pending;
  assert.equal(element('best').textContent, '?', 'old a prediction overwrote the ascender being added for d');
  run('current = null;');
  const pending2 = run('recognize()');
  run('clearAll()');
  resolve(response('d'));
  await pending2;
  assert.equal(element('best').textContent, '?', 'clear did not discard a pending result');
  console.log('Character UI regression checks passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
