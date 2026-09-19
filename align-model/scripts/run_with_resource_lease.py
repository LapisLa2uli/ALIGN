"""Run an arbitrary command while holding an ALIGN resource lease."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from alignmodel.training_resources import resource_lease


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-status", type=Path, required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")
    with resource_lease(
        args.resource_status,
        args.resource,
        track=args.track,
        command=command,
        metadata={"wrapper_pid": __import__("os").getpid()},
    ):
        result = subprocess.run(command, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
