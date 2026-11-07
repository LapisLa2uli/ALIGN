// Exercise the actual browser serialization/undo functions without a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../src/datacreate/web/static/annotate.js'), 'utf8');
function functionSource(name, next) {
  return source.slice(source.indexOf(`function ${name}(`), source.indexOf(`function ${next}(`));
}
const data = {
  id: 'p', source: 'agent', type: 'repetition', extra_copies: 2,
  score_event_indices: [2, 4], context_score_event_indices: [0, 1, 2, 3, 4, 5, 6],
  core_note_ids: ['note_0002', 'note_0004'],
  location_policy: 'stored_transcription_lookup_v1',
  location_evidence: {method: 'covered_transcript_events'},
};
const context = vm.createContext({
  regionsPlugin: {getRegions: () => [{start: 1, end: 2, data}]},
  isTrimRegion: () => false, isLinkOverlay: () => false,
  cloneJson: value => JSON.parse(JSON.stringify(value)),
});
vm.runInContext(functionSource('snapshotRegions', 'snapshotsEqual'), context);
vm.runInContext(functionSource('regionDataToLabel', 'setupRepetitionControls'), context);
const snapshots = vm.runInContext('snapshotRegions()', context);
const label = context.regionDataToLabel(snapshots[0]);
for (const key of ['extra_copies','score_event_indices','context_score_event_indices','location_policy','location_evidence']) {
  assert.deepEqual(JSON.parse(JSON.stringify(label[key])), data[key]);
}
assert.equal(label.start_time, 1);
assert.equal(label.end_time, 2);
const unmapped = context.regionDataToLabel({start:1,end:2,data:{...data,score_event_indices:[],core_note_ids:[]}});
assert.deepEqual(JSON.parse(JSON.stringify(unmapped.score_event_indices)), []);
console.log('Policy 3 GUI serialization and undo checks passed');
