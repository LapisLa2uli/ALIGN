"""Reproduce one failed decode with token-length tracing; preserve frozen outputs."""
import contextlib
import os
import sys
import time
from common import ROOT, RUN, read, write, sha


def main():
    os.environ.update(LADDERSYM_DETERMINISTIC_PROMPT='1', LADDERSYM_PROMPT_LENGTH='1024', LADDERSYM_USE_CACHE='1', PYTHONNOUSERSITE='1')
    import soundfile as sf
    import torch
    from run_model import build
    torch.set_num_threads(3)
    p = read(RUN / 'protocol.json')
    paths = read(RUN / 'data/manifest.json')['tracks']['102']['paths']
    audio, sr = sf.read(paths['mistake_wav'], dtype='float32')
    reference, rr = sf.read(paths['score_wav'], dtype='float32')
    assert sr == rr == 16000
    folder = RUN / 'decoder_trace'
    folder.mkdir(exist_ok=True)
    summaries = {}
    for count in [0, 50]:
        checkpoint = p['initial_checkpoints']['laddersym']['path'] if count == 0 else RUN / 'training/laddersym/n50/model.pt'
        model = build('laddersym', checkpoint).cuda().eval()
        from inference_error import InferenceHandler
        handler = InferenceHandler(model=model, device='cuda', mel_norm=True)
        original = model.generate
        rows = []
        def traced(*args, **kwargs):
            started = time.monotonic()
            result = original(*args, **kwargs)
            tokens = result[0].detach().cpu().tolist()
            rows.append({'window': len(rows), 'output_length': len(tokens),
                'eos_present': model.config.eos_token_id in tokens,
                'tail_tokens': tokens[-32:], 'seconds': time.monotonic() - started})
            return result
        model.generate = traced
        dest = folder / f'laddersym_n{count:02}_102.mid'
        with (folder / f'n{count:02}.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), torch.inference_mode():
            handler.inference(mistake_audio=audio, score_audio=reference, audio_path='102', outpath=dest.as_posix(),
                prompt_path=paths['score_mid'], batch_size=1, max_length=1024)
        frozen = RUN / 'predictions/laddersym' / f'n{count:02}' / '102/mix.mid'
        summaries[str(count)] = {'windows': rows, 'matches_frozen_midi_sha256': sha(dest) == sha(frozen),
            'windows_without_eos': [r['window'] for r in rows if not r['eos_present']]}
        print(count, 'lengths', [r['output_length'] for r in rows], 'without EOS', summaries[str(count)]['windows_without_eos'],
            'same MIDI', summaries[str(count)]['matches_frozen_midi_sha256'], flush=True)
        del handler, model, original
        torch.cuda.empty_cache()
    write(RUN / 'decoder_regression_trace.json', summaries)


if __name__ == '__main__':
    main()
