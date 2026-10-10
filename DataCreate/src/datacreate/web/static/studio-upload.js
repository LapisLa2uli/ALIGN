import {uploadedAudioWav} from './studio-audio.js';
import {createProgressView} from './studio-progress.js';

const $ = id => document.getElementById(id);
const view = createProgressView();
$('progress-screen').append(view.element);
const storageKey = 'align-upload-take';
let jobId = sessionStorage.getItem(storageKey), timer, busy = false, ready = false, maxSeconds = 300;

function show(name) {
  for (const screen of ['upload', 'progress', 'failure', 'success']) $(screen + '-screen').hidden = screen !== name;
  document.title = name === 'success' ? 'feedback uploaded!' : 'Upload performance — ALIGN';
}
function failure(message, retry = false, video = false) {
  busy = false;
  $('failure-message').textContent = message;
  $('retry-feedback').hidden = !retry;
  $('retry-feedback').textContent = video ? 'Retry score animation' : 'Retry feedback';
  show('failure');
}
async function request(url, options) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) {
    const error = new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status}). Please try again.`);
    error.status = response.status; throw error;
  }
  return data;
}
async function settings() {
  try {
    const config = await request('/api/studio/config');
    ready = config.ready; maxSeconds = config.max_seconds;
    $('upload-readiness').textContent = config.message;
    $('score-choice').replaceChildren(new Option('Choose a score', ''), ...config.scores.map(s => new Option(s.name, s.id)));
    if (config.scores.length === 1) $('score-choice').value = config.scores[0].id;
    $('upload-submit').disabled = !ready;
  } catch (error) {
    $('upload-readiness').textContent = error.message;
    $('upload-submit').disabled = true;
  }
}
async function poll() {
  const id = jobId;
  if (!id) return;
  try {
    const state = await request(`/api/studio/takes/${id}`);
    if (id !== jobId) return;
    if (state.status === 'complete') {
      const video = $('feedback-video');
      const videoUrl = state.video_url || state.preview_video_url;
      video.hidden = !videoUrl;
      $('video-unavailable').hidden = !!videoUrl;
      $('video-unavailable').textContent = state.video_message || 'No video was generated for this take, and no matching saved video is available.';
      $('video-reused').hidden = !videoUrl || !!state.video_url;
      if (videoUrl) video.src = videoUrl;
      busy = false; show('success'); return;
    }
    if (state.status === 'failed') {
      failure(state.message, state.can_retry, state.retry_kind === 'video'); return;
    }
    busy = true; show('progress');
    if (state.detailed_progress || state.analysis_progress) view.render(state.detailed_progress, state.analysis_progress);
    else view.waiting(state.message || 'Preparing your take…');
  } catch (error) {
    if (id !== jobId) return;
    if (error.status === 404) {
      sessionStorage.removeItem(storageKey); jobId = null;
      failure('This take is no longer available. Return to upload and submit it again.'); return;
    }
    show('progress'); view.waiting('Connection interrupted. Reconnecting to your take…');
  }
  timer = setTimeout(poll, 2000);
}
$('score-choice').addEventListener('change', () => { $('score-file').value = ''; });
$('score-file').addEventListener('change', () => { if ($('score-file').files.length) $('score-choice').value = ''; });
$('upload-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (busy || !ready) return;
  const audio = $('performance-file').files[0], score = $('score-file').files[0], scoreId = $('score-choice').value;
  $('upload-error').hidden = true;
  try {
    if (!audio?.size || audio.size > 60_000_000) throw new Error('Choose a nonempty audio file smaller than 60 MB.');
    if (!score && !scoreId) throw new Error('Choose or upload the matching score.');
    if (score && (!/\.(musicxml|xml|mxl)$/i.test(score.name) || !score.size || score.size > 10_000_000)) throw new Error('Choose a nonempty MusicXML score smaller than 10 MB.');
    busy = true; show('progress'); view.waiting('Opening and preparing your recording…');
    const decoder = new OfflineAudioContext(1, 1, 48000);
    let decoded;
    try { decoded = await decoder.decodeAudioData(await audio.arrayBuffer()); }
    catch { throw new Error('This file could not be opened as audio. Try WAV or MP3.'); }
    const wav = uploadedAudioWav(decoded, maxSeconds);
    const form = new FormData(); form.append('audio', wav, 'recording.wav');
    if (score) form.append('score', score); else form.append('score_id', scoreId);
    view.waiting('Uploading your recording and score…');
    const job = await request('/api/studio/takes', {method: 'POST', body: form});
    jobId = job.id; sessionStorage.setItem(storageKey, jobId);
    poll();
  } catch (error) {
    busy = false; show('upload'); $('upload-error').textContent = error.message; $('upload-error').hidden = false;
  }
});
$('retry-feedback').addEventListener('click', async () => {
  if (busy || !jobId) return;
  busy = true; $('retry-feedback').disabled = true;
  try {
    await request(`/api/studio/takes/${jobId}/retry`, {method: 'POST'});
    show('progress'); view.waiting('Resuming saved feedback…'); poll();
  } catch (error) { failure(error.message, true); }
  finally { $('retry-feedback').disabled = false; }
});
$('new-upload').addEventListener('click', () => {
  clearTimeout(timer); jobId = null; busy = false; sessionStorage.removeItem(storageKey); show('upload'); settings();
});
window.addEventListener('beforeunload', event => {
  if (busy && !jobId) { event.preventDefault(); event.returnValue = ''; }
});
window.addEventListener('pagehide', () => clearTimeout(timer));
if (jobId) { show('progress'); view.waiting('Reconnecting to your take…'); poll(); }
else { show('upload'); settings(); }
