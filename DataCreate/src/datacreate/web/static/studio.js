import {createProgressView} from './studio-progress.js';
import {melFilters, melEnergy, encodeWav, uploadedAudioWav} from './studio-audio.js';

const $ = id => document.getElementById(id);
let mode = 'idle', stream, context, analyser, recorder, source, mute;
let filters, frequencies, chunks = [], recordingBlob, recordingURL, currentJob;
let started = 0, ticker, stopAcknowledged, maxSeconds = 300, pollTimer;
const canvas = $('spectrum'), pen = canvas.getContext('2d');
const progressView = createProgressView();
const analysisPanel = progressView.element;
document.querySelector('.workspace').before(analysisPanel);
let progressFocused = false;
let display = Array(96).fill(0), width = 1, height = 1;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

function error(message = '') { $('error').textContent = message; $('error').hidden = !message; }
function updateControls() {
  const busy = ['starting', 'stopping', 'decoding', 'uploading', 'processing'].includes(mode);
  $('record').disabled = busy || (!hasScore() && mode !== 'recording');
  $('record-label').textContent = mode === 'recording' ? 'Stop & get feedback' : mode === 'starting' ? 'Opening microphone…' : recordingBlob ? 'Record another take' : 'Start recording';
  $('record').classList.toggle('recording', mode === 'recording');
  $('live-badge').textContent = mode === 'recording' ? '● RECORDING' : mode === 'processing' ? 'PROCESSING' : recordingBlob ? 'TAKE SAVED' : 'STANDBY';
  $('live-badge').classList.toggle('active', mode === 'recording');
  $('score-select').disabled = busy || mode === 'recording';
  $('score-upload').disabled = busy || mode === 'recording';
  $('choose-audio').disabled = busy || mode === 'recording';
  $('audio-upload').disabled = busy || mode === 'recording';
  $('choose-audio').textContent = mode === 'decoding' ? 'Opening audio…' : 'Upload audio ↑';
  $('submit').disabled = busy || mode === 'recording' || !recordingBlob || !hasScore();
}
function hasScore() { return Boolean($('score-select').value || $('score-upload').files.length); }
function setMode(value) { mode = value; updateControls(); }
function formatTime(seconds) { return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`; }
function resetFeedback() {
  progressFocused = false;
  clearTimeout(pollTimer);
  for (const id of ['result', 'transcript', 'retry', 'progress', 'analysis-progress']) $(id).hidden = true;
  $('feedback-playback').pause(); $('feedback-playback').removeAttribute('src');
  $('feedback-video').pause(); $('feedback-video').removeAttribute('src');
  $('feedback-video').hidden = true; $('download-video').hidden = true;
  $('empty-feedback').hidden = false; $('job-status').textContent = '';
  $('feedback-description').textContent = 'Your spoken feedback will be here when your take is ready.';
  sessionStorage.removeItem('align-studio-take'); currentJob = null;
}
async function cleanupAudio() {
  clearInterval(ticker);
  stream?.getTracks().forEach(track => track.stop()); stream = null;
  source?.disconnect(); recorder?.disconnect(); mute?.disconnect();
  if (context && context.state !== 'closed') await context.close();
  context = analyser = recorder = source = mute = null;
}
async function startRecording() {
  error();
  $('feedback-video').pause();
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
    error('Microphone recording needs a current browser on localhost or HTTPS.'); return;
  }
  setMode('starting');
  try {
    $('recording-playback').pause(); $('feedback-playback').pause();
    // Resume on the click gesture before awaiting microphone permission.
    context = new AudioContext({sampleRate: 48000}); await context.resume();
    stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false}});
    await context.audioWorklet.addModule('/static/studio-worklet.js');
    source = context.createMediaStreamSource(stream);
    analyser = context.createAnalyser(); analyser.fftSize = 4096; analyser.smoothingTimeConstant = 0;
    frequencies = new Float32Array(analyser.frequencyBinCount);
    filters = melFilters(context.sampleRate, analyser.fftSize);
    recorder = new AudioWorkletNode(context, 'studio-recorder');
    chunks = [];
    recorder.port.onmessage = ({data}) => {
      if (data === 'stopped') stopAcknowledged?.();
      else if (data === 'limit') { if (mode === 'recording') stopRecording(); }
      else chunks.push(data);
    };
    mute = context.createGain(); mute.gain.value = 0;
    source.connect(analyser); source.connect(recorder); recorder.connect(mute); mute.connect(context.destination);
    resetFeedback(); recordingBlob = null; $('take-preview').hidden = true;
    $('audio-file-name').hidden = true; $('capture-label').textContent = 'MONO · MIC INPUT';
    if (recordingURL) URL.revokeObjectURL(recordingURL);
    started = context.currentTime;
    setMode('recording');
    $('record-hint').textContent = 'Listening. Play at your own pace, then stop to receive your feedback.';
    $('visual-label').textContent = 'YOUR SOUND, TAKING SHAPE';
    ticker = setInterval(() => {
      const seconds = context ? context.currentTime - started : 0;
      $('timer').innerHTML = `${formatTime(seconds)}<span> / ${formatTime(maxSeconds)}</span>`;
      if (seconds >= maxSeconds && mode === 'recording') stopRecording();
    }, 150);
    stream.getAudioTracks()[0].onended = () => { if (mode === 'recording') { error('Microphone disconnected. Your captured audio has been saved.'); stopRecording(); } };
  } catch (failure) {
    await cleanupAudio(); setMode('idle');
    error(failure.name === 'NotAllowedError' ? 'Microphone access was denied. Allow microphone access in your browser, then try again.' : failure.name === 'NotFoundError' ? 'No microphone was found. Connect a microphone and try again.' : 'Could not start the microphone. Check your input device and try again.');
  }
}
async function stopRecording() {
  if (mode !== 'recording') return;
  setMode('stopping'); clearInterval(ticker);
  const rate = context.sampleRate;
  // Flush the partial worklet buffer before closing the input stream.
  await new Promise(resolve => {
    const timeout = setTimeout(resolve, 1500);
    stopAcknowledged = () => { clearTimeout(timeout); resolve(); };
    recorder.port.postMessage('stop');
  });
  stopAcknowledged = null;
  await cleanupAudio();
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  if (total < rate) { chunks = []; setMode('idle'); error('That take was too short. Record at least one second.'); return; }
  // Bound the WAV even if a background-tab timer was throttled.
  let remaining = maxSeconds * rate;
  chunks = chunks.map(chunk => { const kept = chunk.subarray(0, Math.max(0, remaining)); remaining -= kept.length; return kept; });
  recordingBlob = encodeWav(chunks, rate); chunks = [];
  recordingURL = URL.createObjectURL(recordingBlob); $('recording-playback').src = recordingURL;
  $('take-preview').hidden = false;
  $('timer').innerHTML = `${formatTime(Math.min(total / rate, maxSeconds))}<span> / ${formatTime(maxSeconds)}</span>`;
  $('record-hint').textContent = 'Listen back or send your take for analysis and spoken feedback.';
  $('visual-label').textContent = 'A MOMENT OF MUSIC, CAPTURED'; setMode('recorded');
  await submit();
}
async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const failure = new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${response.status}). Please try again.`);
    failure.status = response.status; throw failure;
  }
  return response.json();
}
async function loadAudioFile(file) {
  if (!file) return;
  error();
  if (!file.size || file.size > 60_000_000) { error('Choose a nonempty audio file smaller than 60 MB.'); return; }
  const previousMode = mode;
  setMode('decoding');
  $('recording-playback').pause(); $('feedback-playback').pause();
  try {
    // Offline decoding needs no microphone permission and resamples to 48 kHz.
    const decoder = new OfflineAudioContext(1, 1, 48000);
    let decoded;
    try { decoded = await decoder.decodeAudioData(await file.arrayBuffer()); }
    catch { throw new Error('This file could not be opened as audio. Try WAV or MP3, or export it again from your music app.'); }
    const wav = uploadedAudioWav(decoded, maxSeconds);
    resetFeedback();
    if (recordingURL) URL.revokeObjectURL(recordingURL);
    recordingBlob = wav; recordingURL = URL.createObjectURL(wav);
    $('recording-playback').src = recordingURL; $('take-preview').hidden = false;
    $('audio-file-name').textContent = file.name; $('audio-file-name').hidden = false;
    $('capture-label').textContent = 'UPLOADED AUDIO';
    $('timer').innerHTML = `${formatTime(decoded.duration)}<span> / ${formatTime(maxSeconds)}</span>`;
    $('visual-label').textContent = 'YOUR PERFORMANCE, READY TO LISTEN';
    $('record-hint').textContent = 'Listen back, choose the matching score, then select Get my feedback.';
    setMode('recorded');
  } catch (failure) { setMode(previousMode); error(failure.message); }
}
async function submit() {
  error(); resetFeedback(); setMode('uploading');
  $('recording-playback').pause();
  const data = new FormData(); data.append('audio', recordingBlob, 'recording.wav');
  if ($('score-upload').files[0]) data.append('score', $('score-upload').files[0]);
  else data.append('score_id', $('score-select').value);
  $('job-status').textContent = 'Uploading your take…'; $('empty-feedback').hidden = true;
  showProgressWaiting('Uploading your recording and score…');
  try {
    const job = await request('/api/studio/takes', {method: 'POST', body: data});
    currentJob = job.id; sessionStorage.setItem('align-studio-take', currentJob);
    setMode('processing'); poll();
  } catch (failure) { error(failure.message); $('analysis-title').textContent = 'Upload needs attention'; $('analysis-current').textContent = failure.message; analysisPanel.dataset.status = 'failed'; $('job-status').textContent = 'Your recording is still available above.'; setMode('recorded'); }
}
function showFeedbackMedia(state) {
  const video = $('feedback-video'), audio = $('feedback-playback');
  if (state.video_url) {
    if (video.getAttribute('src') !== state.video_url) video.src = state.video_url;
    $('download-video').href = state.video_url;
    audio.pause();
  }
  video.hidden = !state.video_url;
  $('download-video').hidden = !state.video_url;
  if (state.audio_url) {
    if (audio.getAttribute('src') !== state.audio_url) audio.src = state.audio_url;
    $('download').href = state.audio_url;
  }
  audio.hidden = Boolean(state.video_url) || !state.audio_url;
  $('download').hidden = !state.audio_url;
  $('result').hidden = !state.video_url && !state.audio_url;
}
async function poll() {
  const jobId = currentJob;
  try {
    const state = await request(`/api/studio/takes/${jobId}`);
    if (currentJob !== jobId) return;
    $('job-status').textContent = state.message; $('empty-feedback').hidden = true;
    $('progress').hidden = true;
    renderAnalysisProgress(state.detailed_progress, state.analysis_progress);
    const step = ['queued', 'score', 'reference', 'audio'].includes(state.stage) ? 0 : ['alignment', 'features'].includes(state.stage) ? 1 : 2;
    document.querySelectorAll('.progress li').forEach((item, i) => { item.classList.toggle('current', i === step); item.classList.toggle('done', i < step); });
    if (state.narration) { $('narration').textContent = state.narration; $('transcript').hidden = false; }
    showFeedbackMedia(state);
    if (state.status === 'complete') {
      const details = state.feedback_details;
      $('feedback-description').textContent = details?.no_issues_marked
        ? `${state.assessment?.message || 'The detector marked no specific issues. This does not establish an error-free performance.'} No targeted audio snippets are available.`
        : state.video_url
          ? 'Watch the score animation with the reference, your transcribed playing, and practice advice.'
        : details?.reference_excerpts
          ? 'Listen to the reference, your performance, then the correction and practice advice.'
          : details?.performance_excerpts
            ? 'Listen to the marked passages from your performance, followed by practice advice.'
            : 'Press play. Take one idea into your next practice session.';
      if (state.video_message) $('feedback-description').textContent += ` ${state.video_message}`;
      setMode(recordingBlob ? 'recorded' : 'idle'); return;
    }
    if (state.status === 'failed') {
      $('retry').hidden = !state.can_retry;
      $('retry').textContent = state.retry_kind === 'video' ? 'Retry score animation' : 'Retry spoken feedback';
      $('transcript').hidden = !state.narration;
      $('feedback-description').textContent = state.assessment?.status === 'alignment_uncertain'
        ? 'Analysis was inconclusive. Check the matching score before submitting again.'
        : state.retry_kind === 'video' ? 'Your feedback audio is ready below. Retry to finish the score animation.'
        : state.narration ? 'Your written feedback is saved. Retry to create the audio.' : 'This take needs a little attention.';
      setMode(recordingBlob ? 'recorded' : 'idle'); return;
    }
    setMode('processing');
  } catch (failure) {
    if (currentJob !== jobId) return;
    if (failure.status === 404) {
      resetFeedback(); setMode(recordingBlob ? 'recorded' : 'idle');
      error('This take is no longer available on the server. Record or submit a new take.'); return;
    }
    $('job-status').textContent = 'Connection interrupted. Reconnecting to your take…';
    $('analysis-current').textContent = 'Connection interrupted. Reconnecting to your take…';
    progressView.waiting('Connection interrupted. Reconnecting to your take…');
  }
  pollTimer = setTimeout(poll, 2000);
}
function focusProgress() {
  if (!progressFocused) {
    progressFocused = true;
    requestAnimationFrame(() => analysisPanel.scrollIntoView({behavior: reducedMotion ? 'auto' : 'smooth',
      block: analysisPanel.getBoundingClientRect().height > innerHeight - 48 ? 'start' : 'center'}));
  }
}
function showProgressWaiting(message) {
  progressView.waiting(message);
  focusProgress();
}
function renderAnalysisProgress(progress, legacy) {
  progressView.render(progress, legacy);
  if (progress?.status === 'processing') focusProgress();
}
$('record').addEventListener('click', () => mode === 'recording' ? stopRecording() : startRecording());
$('submit').addEventListener('click', submit);
$('choose-audio').addEventListener('click', () => $('audio-upload').click());
$('audio-upload').addEventListener('change', async () => {
  await loadAudioFile($('audio-upload').files[0]);
  $('audio-upload').value = ''; // Allow choosing the same file again after a failure.
});
$('retry').addEventListener('click', async () => {
  progressFocused = false;
  error(); $('retry').disabled = true;
  try { await request(`/api/studio/takes/${currentJob}/retry`, {method: 'POST'}); $('retry').hidden = true; setMode('processing'); poll(); }
  catch (failure) { error(failure.message); }
  finally { $('retry').disabled = false; }
});
$('score-select').addEventListener('change', () => { $('score-upload').value = ''; $('upload-label').textContent = 'Or upload a score'; updateControls(); });
$('score-upload').addEventListener('change', () => {
  const file = $('score-upload').files[0];
  if (file && (!/\.(musicxml|xml|mxl)$/i.test(file.name) || file.size > 10_000_000)) { error('Choose a MusicXML score smaller than 10 MB.'); $('score-upload').value = ''; return; }
  if (file) { $('score-select').value = ''; $('upload-label').textContent = file.name; }
  updateControls();
});
window.addEventListener('beforeunload', event => {
  if (['recording', 'starting', 'stopping', 'decoding', 'uploading'].includes(mode) || (recordingBlob && !currentJob)) { event.preventDefault(); event.returnValue = ''; }
});
window.addEventListener('pagehide', () => { stream?.getTracks().forEach(track => track.stop()); context?.close(); });
new ResizeObserver(() => {
  const bounds = canvas.getBoundingClientRect(), ratio = Math.min(devicePixelRatio || 1, 2);
  width = bounds.width; height = bounds.height; canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); pen.setTransform(ratio, 0, 0, ratio, 0, 0);
}).observe(canvas);
function draw(time) {
  pen.clearRect(0, 0, width, height);
  let energy = Array(96).fill(0);
  if (analyser && mode === 'recording') { analyser.getFloatFrequencyData(frequencies); energy = melEnergy(frequencies, filters); }
  const gap = width / 112, barWidth = Math.max(2, gap * .48), middle = height * .46;
  pen.strokeStyle = '#e7ecdf'; pen.lineWidth = 1;
  pen.beginPath(); pen.moveTo(10, middle); pen.lineTo(width - 10, middle); pen.stroke();
  for (let i = 0; i < 96; i++) {
    display[i] += (energy[i] - display[i]) * .22;
    const envelope = Math.sin(Math.PI * i / 95) ** 1.2;
    const idle = mode === 'recording' ? 0 : envelope * (13 + 8 * Math.sin(i * .19) ** 2 + (reducedMotion ? 0 : 3 * Math.sin(time / 1300 + i * .12)));
    const amplitude = 2 + idle + display[i] * (height * .34) * (.3 + .7 * envelope);
    const x = (width - 95 * gap) / 2 + i * gap;
    pen.fillStyle = i < 50 ? `rgba(85,126,87,${.35 + envelope * .4})` : `rgba(143,157,91,${.4 + envelope * .4})`;
    pen.beginPath(); pen.roundRect(x, middle - amplitude, barWidth, amplitude * 2, barWidth / 2); pen.fill();
  }
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
try {
  const settings = await request('/api/studio/config');
  maxSeconds = settings.max_seconds;
  $('instrument').textContent = settings.instrument; $('service-status').textContent = settings.message;
  settings.scores.forEach(score => { const option = document.createElement('option'); option.value = score.id; option.textContent = score.name; $('score-select').append(option); });
  if (settings.scores.length === 1) $('score-select').value = settings.scores[0].id;
  currentJob = sessionStorage.getItem('align-studio-take');
  if (currentJob) { setMode('processing'); poll(); } else updateControls();
} catch (failure) { $('service-status').textContent = 'Studio unavailable. Start the DataCreate server and reload.'; error(failure.message); }
