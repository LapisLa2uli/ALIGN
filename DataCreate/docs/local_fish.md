# Fish Speech running locally on Windows

This installation uses the official **Fish Speech 1.5** weights and **v1.5.1**
inference code, suitable for the NVIDIA RTX 2000 Ada 8 GB GPU. It is a different,
older model than the hosted S2.1-Pro API. No Fish account or API key is required.
The weights use **CC-BY-NC-SA-4.0**, so this setup is for non-commercial use.
See the downloaded model README for the license link and attribution.

Everything lives in `.local/fish/` (ignored by Git): an isolated Python 3.11
environment, official source, downloaded weights, and server logs. Existing ALIGN
training environments are not modified. We use a small FastAPI wrapper around
the upstream TTS engine, avoiding unrelated ASR, training, and web UI packages.
Compilation is disabled for native Windows compatibility.

## Start

From the project root in PowerShell:

```powershell
./DataCreate/scripts/start_fish_local.ps1
Invoke-RestMethod http://127.0.0.1:8081/v1/health
```

The process starts hidden and binds only `127.0.0.1:8081`. Logs are in
`.local/fish/server.stdout.log` and `server.stderr.log`, with PID in `server.pid`.
Use `-Foreground` for visible terminal logs and Ctrl+C to stop. For a background
process, check that the PID still belongs to the Fish Python server before using
`Stop-Process -Id <pid>`. There is no automatic startup when Windows restarts.

The model stays loaded on the GPU until the service exits. Stop it before large
ALIGN training jobs if they need the same VRAM. Requests run one at a time;
another concurrent synthesis gets HTTP 409. Long narration can take minutes.

## Produce an MP3

Use the existing feedback environment/entry point:

```powershell
# Entirely local speech from a saved narration; no external APIs or keys.
datacreate-feedback --text path/to/feedback.txt `
  --config DataCreate/config/feedback.local.yaml --output feedback/local-take

# LLM analysis through 302.AI followed by local Fish speech.
$env:API_302_KEY = 'your-key'
datacreate-feedback --labels DataCreate/samples/020/labels_agent.json `
  --config DataCreate/config/feedback.local.yaml --output feedback/local-analysis
```

The text-only input path bypasses the LLM. The label input path still sends the
normalized report to 302.AI; only speech synthesis is local. The service sets
Hugging Face and Transformers offline mode after installation. Output is a real
MP3 encoded locally with libsndfile; FFmpeg is not required.

The wrapper uses Fish's 300-byte text chunk setting, keeping short feedback
together when possible. This avoids the previous split immediately after
`bar 2` in the sample 001 narration. Longer feedback may still span chunks;
listen to generated speech to check pronunciation and transitions.

Without a reference voice, Fish chooses a voice; timbre may vary. To select a
consistent voice, place your reference audio and matching `.lab` transcript
under `.local/fish/fish-speech-1.5.1/references/<voice-name>/`, then set
`fish_local_reference_id: <voice-name>` in the local config. Hosted Fish voice
IDs do not automatically resolve to local audio. Use audio you have permission
to use, ideally a clear short spoken reference.

### Human-reference voice trial

`DataCreate/config/feedback.ssstoken.teacher.yaml` selects `teacher_lj`, an
11.55-second human narration reference assembled from consecutive public-domain
LJSpeech clips `LJ001-0001` and `LJ001-0002`. The audio, matching `.lab` transcript,
source URLs, and checksums are stored under
`.local/fish/fish-speech-1.5.1/references/teacher_lj/`. See the
[dataset provenance and license](https://keithito.com/LJ-Speech-Dataset/).
This is an audiobook narrator used to test a teacher-like delivery, not a
dedicated trained teacher voice. Listen to the result before choosing it as the
default; perceived naturalness is not guaranteed by reference conditioning.

Reuse saved feedback to compare voices without paying for another LLM call:

```powershell
datacreate-feedback --text path/to/feedback.txt `
  --config DataCreate/config/feedback.ssstoken.teacher.yaml `
  --output feedback/teacher-voice-trial
```

For label input, this trial config uses `gpt-6-luna` at
`https://api.ssstoken.net/v1` and expects its key in `OPENAI_API_KEY`.
Reference assets are local and ignored by Git; another installation needs its
own paired reference files before using this config.

## Reinstall

Requires an NVIDIA driver, internet access for downloads, Python 3.11, and several
GB of free disk space. The runtime and large weights are not committed to Git.

```powershell
./DataCreate/scripts/install_fish_local.ps1 -Python path/to/python.exe
```

Source tag v1.5.1 resolves to commit
`58046eaa1a4cefb0c8cc3a3a667b34186ea02dde`.
Weights are pinned to Hugging Face revision
`275a984d33c33659e39eed41ff5bcd6e67517f4c`; the downloader verifies both model
and codec SHA-256 values and writes a manifest for all downloaded assets.

- [Official source](https://github.com/fishaudio/fish-speech/tree/v1.5.1)
- [Official weights and license](https://huggingface.co/fishaudio/fish-speech-1.5)
