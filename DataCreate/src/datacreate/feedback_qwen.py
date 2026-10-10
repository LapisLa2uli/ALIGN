"""Beijing Qwen-Audio HTTP TTS; no SDK or local GPU required.

Uses the SpeechSynthesizer protocol, not the older Qwen3-TTS protocol.
API reference: https://help.aliyun.com/en/model-studio/qwen-audio-tts-http-api
"""
from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

import httpx

from datacreate.feedback import FeedbackConfig, FeedbackError, _check_status, _secret, _write_json


def qwen_endpoint(config: FeedbackConfig) -> str:
    workspace = _secret(config.qwen_workspace_id_env)
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,62})", workspace):
        raise FeedbackError("DASHSCOPE_WORKSPACE_ID must be the Beijing workspace ID, not a URL.")
    return f"https://{workspace}.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"


def _audio_url(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise FeedbackError("Qwen Audio returned no audio URL.")
    parsed = urlsplit(value)
    # The official response example uses HTTP for signed OSS URLs. Upgrade to
    # HTTPS without modifying its path/query; never follow arbitrary redirects.
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or not parsed.hostname.endswith(".aliyuncs.com")
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.fragment):
        raise FeedbackError("Qwen Audio returned an unsupported audio URL.")
    return urlunsplit(parsed._replace(scheme="https"))


def synthesize_qwen(text: str, output: Path, config: FeedbackConfig, client: httpx.Client) -> None:
    endpoint = qwen_endpoint(config)
    temporary = output.with_suffix(".mp3.part")
    try:
        response = client.post(endpoint, headers={"Authorization": "Bearer " + _secret(config.qwen_api_key_env)},
                               json={"model": config.qwen_model, "input": {
                                   "text": text, "voice": config.qwen_voice, "format": "mp3",
                                   "sample_rate": 44100, "rate": config.qwen_rate,
                                   "instruction": config.qwen_instruction,
                               }}, follow_redirects=False)
        _check_status(response, "Qwen Audio")
        data = response.json()
        if not isinstance(data, dict) or data.get("code") or data["output"]["finish_reason"] != "stop":
            raise FeedbackError("Qwen Audio did not complete synthesis. Check model/voice access and quota.")
        url = _audio_url(data["output"]["audio"]["url"])
        # Signed OSS downloads must never receive the synthesis key or cookies,
        # including defaults supplied by a caller's httpx client.
        request = client.build_request("GET", url)
        for header in ("authorization", "cookie"):
            request.headers.pop(header, None)
        audio = client.send(request, stream=True, auth=None, follow_redirects=False)
        try:
            _check_status(audio, "Qwen audio download")
            mime = audio.headers.get("content-type", "").split(";")[0].lower()
            if mime not in {"audio/mpeg", "audio/mp3", "application/octet-stream"}:
                raise FeedbackError("Qwen Audio returned a non-MP3 download.")
            size = 0
            with temporary.open("wb") as stream:
                for chunk in audio.iter_bytes():
                    size += len(chunk)
                    if size > 50_000_000:
                        raise FeedbackError("Qwen Audio response exceeded the 50 MB audio limit.")
                    stream.write(chunk)
            with temporary.open("rb") as stream:
                header = stream.read(3)
            if size < 4 or not (header == b"ID3" or (header[0] == 0xFF and header[1] & 0xE0 == 0xE0)):
                raise FeedbackError("Qwen Audio did not return an MP3 header.")
        finally:
            audio.close()
        # Preserve token counts for cost inspection, never URLs or provider text.
        usage = data.get("usage", {})
        usage = {key: usage[key] for key in ("input_tokens", "output_tokens", "total_tokens", "characters")
                 if isinstance(usage, dict) and type(usage.get(key)) is int and usage[key] >= 0}
        _write_json(output.with_suffix(".tts.json"), {"provider": "qwen", "model": config.qwen_model,
                    "voice": config.qwen_voice, "usage": usage})
        temporary.replace(output)
    except httpx.HTTPError:
        raise FeedbackError("Qwen Audio network request failed. Saved narration and snippets are preserved. "
                            "Retry with --plan (when available) or --text. No automatic retry was made.") from None
    except (KeyError, IndexError, TypeError, ValueError) as error:
        if isinstance(error, FeedbackError):
            raise
        raise FeedbackError("Qwen Audio returned an invalid synthesis response.") from None
    finally:
        temporary.unlink(missing_ok=True)
