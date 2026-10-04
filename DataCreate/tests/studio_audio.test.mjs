import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
// Load the browser ES module without changing the existing project's package type.
const code = await readFile(new URL('../src/datacreate/web/static/studio-audio.js', import.meta.url), 'utf8');
const {melFilters, melEnergy, encodeWav, uploadedAudioWav} = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('uploaded stereo audio is converted to mono PCM for the pipeline', async () => {
  const channels = [new Float32Array([1, -.5, .5]), new Float32Array([0, .5, .5])];
  const wav = uploadedAudioWav({duration: 1, length: 3, sampleRate: 48000, numberOfChannels: 2, getChannelData: i => channels[i]});
  const view = new DataView(await wav.arrayBuffer());
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 48000);
  assert.deepEqual(Array.from({length: 3}, (_, i) => view.getInt16(44 + i * 2, true)), [16383, 0, 16383]);
});

test('uploads reject short, overlong and invalid durations without truncation', () => {
  for (const duration of [0, .5, 301, NaN, Infinity]) {
    assert.throws(() => uploadedAudioWav({duration}), /between 1 second/);
  }
});

test('mel filter bank places a tone in its corresponding frequency band', () => {
  const filters = melFilters(48000, 4096);
  const silence = new Float32Array(2048).fill(-Infinity);
  assert.ok(melEnergy(silence, filters).every(value => value === 0));
  function peak(hz) {
    const spectrum = silence.slice(); spectrum[Math.round(hz * 4096 / 48000)] = -15;
    const energy = melEnergy(spectrum, filters);
    assert.ok(energy.every(Number.isFinite));
    assert.ok(Math.max(...energy) > .5);
    return energy.indexOf(Math.max(...energy));
  }
  assert.ok(peak(440) < peak(1760));
});

test('PCM WAV preserves sample rate, lengths, clipping and signed samples', async () => {
  const wav = encodeWav([new Float32Array([-2, -.5]), new Float32Array([0, .5, 2])], 48000);
  const view = new DataView(await wav.arrayBuffer());
  assert.equal(wav.type, 'audio/wav'); assert.equal(view.byteLength, 54);
  assert.equal(view.getUint32(24, true), 48000); assert.equal(view.getUint32(40, true), 10);
  assert.equal(view.getUint16(22, true), 1);
  assert.deepEqual(Array.from({length: 5}, (_, i) => view.getInt16(44 + i * 2, true)), [-32768, -16384, 0, 16383, 32767]);
});

test('worklet downmixes, flushes once and stops at its own capture limit', async () => {
  const messages = [];
  let Recorder;
  const source = await readFile(new URL('../src/datacreate/web/static/studio-worklet.js', import.meta.url), 'utf8');
  vm.runInNewContext(source, {
    AudioWorkletProcessor: class { constructor() { this.port = {postMessage: value => messages.push(value)}; } },
    sampleRate: 8000, Float32Array,
    registerProcessor: (_, implementation) => { Recorder = implementation; },
  });
  const recorder = new Recorder(); recorder.limit = 3;
  const input = [[new Float32Array([1, 1, 1, 1]), new Float32Array([0, 0, 0, 0])]];
  recorder.process(input); recorder.process(input);
  assert.deepEqual([...messages[0]], [.5, .5, .5]); assert.equal(messages[1], 'limit');
  recorder.port.onmessage({data: 'stop'});
  assert.equal(messages[2], 'stopped'); assert.equal(messages.length, 3);
});
