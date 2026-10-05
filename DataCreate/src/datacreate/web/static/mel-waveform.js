import { melFilters, melEnergy } from './studio-audio.js';

/** Attach once per <audio> element; change audio.src to play another file.
 * Call resume() from a click before playback, and destroy() when removing it.
 * Remote audio needs CORS permission and audio.crossOrigin = 'anonymous'
 * set before assigning its URL. Local File object URLs do not need CORS.
 */
export function createMelWaveform(audio, canvas) {
  const context = new AudioContext();
  const source = context.createMediaElementSource(audio);
  const analyser = context.createAnalyser();
  analyser.fftSize = 4096;
  analyser.smoothingTimeConstant = 0;
  source.connect(analyser);
  analyser.connect(context.destination); // Keep the audio audible.

  const filters = melFilters(context.sampleRate, analyser.fftSize, 96);
  const frequencies = new Float32Array(analyser.frequencyBinCount);
  const display = new Float32Array(filters.length);
  const pen = canvas.getContext('2d');
  let width = 1, height = 1, frame, previousTime = 0;

  const resize = new ResizeObserver(() => {
    const bounds = canvas.getBoundingClientRect();
    const ratio = Math.min(devicePixelRatio || 1, 2);
    width = bounds.width; height = bounds.height;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    pen.setTransform(ratio, 0, 0, ratio, 0, 0);
  });
  resize.observe(canvas);

  function draw(time) {
    const playing = !audio.paused && !audio.ended && context.state === 'running';
    let energy = null;
    if (playing) {
      analyser.getFloatFrequencyData(frequencies);
      energy = melEnergy(frequencies, filters);
    }
    // Frame-rate-independent smoothing; the studio's .22 at 60 Hz.
    const delta = previousTime ? Math.min(100, time - previousTime) : 1000 / 60;
    const smoothing = 1 - Math.pow(.78, delta / (1000 / 60));
    previousTime = time;
    pen.clearRect(0, 0, width, height);
    const gap = width / 112, barWidth = Math.max(2, gap * .48);
    const middle = height / 2;
    pen.strokeStyle = '#e7ecdf';
    pen.beginPath(); pen.moveTo(10, middle); pen.lineTo(width - 10, middle); pen.stroke();

    for (let i = 0; i < display.length; i++) {
      display[i] += ((energy?.[i] ?? 0) - display[i]) * smoothing;
      const envelope = Math.sin(Math.PI * i / (display.length - 1)) ** 1.2;
      const amplitude = 2 + display[i] * height * .34 * (.3 + .7 * envelope);
      const x = (width - (display.length - 1) * gap) / 2 + i * gap;
      pen.fillStyle = i < 50
        ? `rgba(85,126,87,${.35 + envelope * .4})`
        : `rgba(143,157,91,${.4 + envelope * .4})`;
      pen.beginPath();
      pen.roundRect(x, middle - amplitude, barWidth, amplitude * 2, barWidth / 2);
      pen.fill();
    }
    frame = requestAnimationFrame(draw);
  }
  frame = requestAnimationFrame(draw);

  return {
    resume: () => context.resume(),
    async destroy() {
      audio.pause();
      cancelAnimationFrame(frame);
      resize.disconnect();
      source.disconnect(); analyser.disconnect();
      await context.close();
    },
  };
}
