class StudioRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.recording = true;
    this.buffer = new Float32Array(2048);
    this.used = 0;
    this.samples = 0;
    this.limit = sampleRate * 300;
    this.port.onmessage = ({data}) => {
      if (data === 'stop') {
        this.recording = false;
        this.flush();
        this.port.postMessage('stopped');
      }
    };
  }
  flush() {
    if (this.used) this.port.postMessage(this.buffer.slice(0, this.used));
    this.used = 0;
  }
  process(inputs) {
    const channels = inputs[0];
    if (this.recording && channels.length) {
      for (let i = 0; i < channels[0].length; i++) {
        let sample = 0;
        for (const channel of channels) sample += channel[i] / channels.length;
        this.buffer[this.used++] = sample;
        this.samples++;
        if (this.used === this.buffer.length) {
          this.port.postMessage(this.buffer);
          this.buffer = new Float32Array(2048); this.used = 0;
        }
        if (this.samples >= this.limit) {
          this.recording = false; this.flush(); this.port.postMessage('limit'); break;
        }
      }
    }
    return true;
  }
}
registerProcessor('studio-recorder', StudioRecorder);
