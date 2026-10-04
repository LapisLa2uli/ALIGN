"""Download pinned official Fish 1.5 assets; no Hugging Face credential required."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

REVISION = "275a984d33c33659e39eed41ff5bcd6e67517f4c"
FILES = {
    "model.pth": "918dc960372cc1b77bbafb14c48ef7a1634ecf75d4eb85b78607223b780d6001",
    "firefly-gan-vq-fsq-8x1024-21hz-generator.pth": "01b81dbf753224a156c3fe139b88bf0b9a0f54b11bee864f95e66511c3ccd754",
    "config.json": None, "special_tokens.json": None, "tokenizer.tiktoken": None, "README.md": None,
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def download(item):
        name, expected = item
        path = args.output / name
        if path.is_file() and expected and digest(path) == expected:
            print(f"Verified existing {name}", flush=True)
            return name, expected
        temporary = path.with_suffix(path.suffix + ".part")
        url = f"https://huggingface.co/fishaudio/fish-speech-1.5/resolve/{REVISION}/{name}"
        print(f"Downloading {name}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        actual = digest(temporary)
        if expected and expected != actual:
            raise RuntimeError(f"Hash mismatch: {name}")
        temporary.replace(path)
        print(f"Verified {name}: {path.stat().st_size:,} bytes", flush=True)
        return name, actual

    with ThreadPoolExecutor(max_workers=2) as pool:
        hashes = dict(pool.map(download, FILES.items()))
    (args.output / "download-manifest.json").write_text(json.dumps({
        "repository": "fishaudio/fish-speech-1.5", "revision": REVISION,
        "sha256": hashes, "weights_license": "CC-BY-NC-SA-4.0",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
