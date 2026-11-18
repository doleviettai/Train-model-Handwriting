const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/ghep/serve_ghep.py', 'utf8');
const html = source.match(/HTML = r"""([\s\S]*?)"""/)[1];
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1].replace('__CLASSES__', '["uê","iê"]');
const elements = new Map();
const ctx = new Proxy({}, {get: () => () => {}});
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    listeners: {}, value: '', checked: true, style: {}, textContent: '', innerHTML: '', disabled: false,
    addEventListener(name, callback) {this.listeners[name] = callback;},
    getContext() {return ctx;},
    getBoundingClientRect() {return {left: 0, top: 0, width: 900, height: 360};},
    setPointerCapture() {},
  });
  return elements.get(id);
}
const sandbox = {document: {getElementById: element, addEventListener() {}},
  window: {devicePixelRatio: 1, addEventListener() {}}, setTimeout() {return 1;}, clearTimeout() {}, console};
vm.createContext(sandbox);
vm.runInContext(script, sandbox);
const run = code => vm.runInContext(code, sandbox);
const response = text => ({ok: true, json: async () => ({text, confidence: 0.8,
  candidates: [{text, p: 0.8}], preview: 'data:image/png;base64,', uncertain: false})});

(async () => {
  let resolve;
  sandbox.fetch = () => new Promise(r => {resolve = r;});
  run('strokes = [[[0,0],[1,1]]]; strokeVersion = 1;');
  const preview = run('preview()');
  element('pad').listeners.pointerdown({clientX: 2, clientY: 2, pointerId: 1});
  resolve(response('uê'));
  await preview;
  assert.equal(run('lastResult'), null, 'old uê survived adding another stroke');

  let calls = 0;
  sandbox.fetch = async (url, options) => {
    calls++;
    assert.equal(JSON.parse(options.body).expected, undefined);
    return response('iê');
  };
  run("current = null; lastResult = {text:'uê',confidence:0.8,version:strokeVersion-1};");
  await Promise.all([run('commitWord()'), run('commitWord()')]);
  assert.equal(calls, 1);
  assert.equal(run('words.length'), 1);
  assert.equal(run('words[0].text'), 'iê');
  sandbox.fetch = () => new Promise(r => {resolve = r;});
  run('strokes = [[[0,0],[1,1]]]; strokeVersion++;');
  const pending = run('commitWord()');
  run('clearCurrent()');
  resolve(response('uê'));
  await pending;
  assert.equal(run('words.length'), 1);
  assert.equal(element('btnNext').disabled, false);
  let savedPayload = null;
  run('strokes = [[[0,0],[1,1]]]; current = null;');
  element('correctLabel').value = '';
  element('writerId').value = '';
  sandbox.fetch = async (url, options) => {
    assert.equal(url, '/samples');
    savedPayload = JSON.parse(options.body);
    return {ok:true, json:async () => ({message:'Saved for the next training run'})};
  };
  await element('btnSaveSample').onclick();
  assert.equal(savedPayload, null, 'a correction was saved without a chosen label/writer');
  element('correctLabel').value = 'iê';
  element('writerId').value = 'tester01';
  await element('btnSaveSample').onclick();
  assert.equal(savedPayload.truth, 'iê');
  assert.equal(savedPayload.writer, 'tester01');
  assert.equal(savedPayload.expected, undefined);
  assert.equal(element('btnSaveSample').disabled, false);
  console.log('Ghep UI regression checks passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
