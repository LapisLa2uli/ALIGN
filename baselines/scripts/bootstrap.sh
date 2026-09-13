#!/usr/bin/env bash
# Rebuild pinned upstream repositories + complete patches + ALIGN configs.
# Usage: bootstrap.sh [--repos-only] [--device cu121|cu118|cpu|mac]
set -euo pipefail
B=${BASELINE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${BASELINE_SYSTEM_PYTHON:-python3.11}
REPOS_ONLY=0
DEVICE=cu121
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repos-only) REPOS_ONLY=1; shift ;;
    --device) DEVICE=$2; shift 2 ;;
    *) echo "usage: $0 [--repos-only] [--device cu121|cu118|cpu|mac]" >&2; exit 2 ;;
  esac
done
case "$DEVICE" in cu121|cu118|cpu|mac) ;; *) echo "Unknown wheel profile: $DEVICE" >&2; exit 2 ;; esac
install_packages() {
  if [[ ${BASELINE_INSTALLER:-auto} != pip ]] && command -v uv >/dev/null 2>&1; then
    uv pip install --python "$py" "$@"
  else
    "$py" -m pip install "$@"
  fi
}
for name in Polytune LadderSym; do
  if [[ "$name" == Polytune ]]; then
    flavor=polytune; commit=d2055bb21759d457c8f21c1cf2e47c79af6248f5
  else
    flavor=laddersym; commit=381179754cf6bcb435f9decf0d5e24eada6c68ec
  fi
  repo="$B/$name"
  if [[ ! -e "$repo" ]]; then
    git clone "https://github.com/ben2002chou/$name" "$repo"
    git -C "$repo" checkout --detach "$commit"
  fi
  [[ $(git -C "$repo" rev-parse HEAD) == "$commit" ]] || {
    echo "$repo is not pinned to $commit; move it aside before bootstrap" >&2; exit 2;
  }
  if git -C "$repo" apply --reverse --check "$B/patches/$flavor.patch" 2>/dev/null; then
    echo "$name patch already applied"
  elif git -C "$repo" diff --quiet && git -C "$repo" apply --check "$B/patches/$flavor.patch"; then
    git -C "$repo" apply "$B/patches/$flavor.patch"
  else
    echo "$name has conflicting edits; no files were reset" >&2; exit 2
  fi
  # Do not overwrite a locally edited ALIGN config.
  while IFS= read -r -d '' src; do
    rel=${src#"$B/configs/$name/"}; dst="$repo/config/$rel"
    mkdir -p "$(dirname "$dst")"
    if [[ -f "$dst" ]] && ! cmp -s "$src" "$dst"; then
      echo "Config differs: $dst; preserve your edits and reconcile with $src" >&2; exit 2
    fi
    cp "$src" "$dst"
  done < <(find "$B/configs/$name" -type f -name '*.yaml' -print0)
  (( REPOS_ONLY )) && continue
  "$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Use Python 3.11"'
  env_dir="$B/envs/$flavor"
  if [[ ! -d "$env_dir" ]]; then "$PYTHON" -m venv "$env_dir"; fi
  py="$env_dir/bin/python"
  "$py" -m ensurepip --upgrade
  install_packages 'pip>=24,<26' 'setuptools==80.10.2' wheel
  torch_args=()
  [[ "$DEVICE" == mac ]] || torch_args+=(--index-url "https://download.pytorch.org/whl/$DEVICE")
  install_packages torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 ${torch_args[@]+"${torch_args[@]}"}
  install_packages -r "$B/envs/requirements.$flavor.server.txt" -c "$B/envs/requirements.$flavor.server.lock.txt"
  "$py" -m pip check
  MPLBACKEND=Agg "$py" "$B/common/doctor.py" --device "$DEVICE"
  "$py" -m pip freeze > "$env_dir/installed.freeze.txt"
done
