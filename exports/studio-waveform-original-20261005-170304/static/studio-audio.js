// A triangular mel filter bank over the analyser's linear FFT power spectrum.
export function melFilters(sampleRate, fftSize, count = 96) {
  const mel = hz => 2595 * Math.log10(1 + hz / 700);
  const hz = value => 700 * (10 ** (value / 2595) - 1);
  const low = mel(30), high = mel(Math.min(12000, sampleRate / 2));
  const edges = Array.from({length: count + 2}, (_, i) => hz(low + (high - low) * i / (count + 1)));
  return Array.from({length: count}, (_, i) => {
    const weights = [];
    for (let bin = Math.ceil(edges[i] * fftSize / sampleRate); bin < fftSize / 2 && bin * sampleRate / fftSize < edges[i + 2]; bin++) {
      const frequency = bin * sampleRate / fftSize;
      const weight = Math.max(0, Math.min((frequency - edges[i]) / (edges[i + 1] - edges[i]), (edges[i + 2] - frequency) / (edges[i + 2] - edges[i + 1])));
      weights.push([bin, weight]);
    }
    return weights;
  });
}

export function melEnergy(decibels, filters) {
  return filters.map(filter => {
    let power = 0, weightSum = 0;
    for (const [bin, weight] of filter) {
      power += 10 ** (decibels[bin] / 10) * weight;
      weightSum += weight;
    }
    const db = 10 * Math.log10(Math.max(1e-12, power / (weightSum || 1)));
    return Math.max(0, Math.min(1, (db + 85) / 65));
  });
}

export function encodeWav(chunks, sampleRate) {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const buffer = new ArrayBuffer(44 + length * 2), view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((character, i) => view.setUint8(offset + i, character.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, 36 + length * 2, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, length * 2, true);
  let offset = 44;
  for (const chunk of chunks) for (const sample of chunk) {
    const value = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, value < 0 ? value * 32768 : value * 32767, true); offset += 2;
  }
  return new Blob([buffer], {type: 'audio/wav'});
}

// Uploaded files are decoded by the browser, then use the same PCM input as mic takes.
export function uploadedAudioWav(buffer, maxSeconds = 300) {
  if (!Number.isFinite(buffer.duration) || buffer.duration < 1 || buffer.duration > maxSeconds) {
    throw new Error(`Choose audio between 1 second and ${maxSeconds / 60} minutes long.`);
  }
  if (!buffer.numberOfChannels || !buffer.length) throw new Error('This audio file is empty.');
  const mono = new Float32Array(buffer.length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
    const samples = buffer.getChannelData(channel);
    for (let i = 0; i < mono.length; i++) mono[i] += samples[i] / buffer.numberOfChannels;
  }
  return encodeWav([mono], buffer.sampleRate);
}
