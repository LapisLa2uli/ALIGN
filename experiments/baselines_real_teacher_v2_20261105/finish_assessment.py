"""Finish the declared assessments after the existing GPU campaign completes.

This runner changes no training, decoding, selection, or scoring decisions.
"""
import os
import subprocess
import sys
import time

from common import RUN, read, sha, write


def main():
    began = time.monotonic()
    while True:
        progress = read(RUN / "progress.json")
        audit_path = RUN / "final_audit.json"
        if progress.get("stage") == "complete" and audit_path.exists():
            audit = read(audit_path)
            if audit.get("status") == "passed" and audit.get("runtime_amendment_verified"):
                break
        if time.monotonic() - began > 4 * 3600:
            raise TimeoutError("Main campaign did not finish within four hours")
        time.sleep(10)

    commands = [
        ["assess_trained.py", "run"],
        ["note_unit_assessment.py", "run"],
        ["analyze_outputs.py", "--phase", "both"],
        ["report_interpretation.py"],
        ["audit_final.py"],
    ]
    for command in commands:
        write(RUN / "completion_progress.json", {"stage": command[0]})
        print("START", *command, flush=True)
        subprocess.run(
            [sys.executable, str(RUN / command[0]), *command[1:]],
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
            check=True,
        )
        print("DONE", *command, flush=True)

    amendment = read(RUN / "runtime_amendment.json")
    for name, digest in amendment["source_sha256"].items():
        assert sha(RUN / name) == digest, name
    assert sha(RUN / "cache_parity.json") == amendment["parity_report_sha256"]
    assert sha(RUN / "protocol.json") == amendment["original_protocol_sha256"]
    trained = read(RUN / "trained_assessment_audit.json")
    assert trained["status"] == "passed"
    assert sha(RUN / "trained_results.json") == trained["results_sha256"]
    assert sha(RUN / "note_unit_assessment.py") == read(RUN / "note_unit_protocol.json")["source_sha256"]
    final = read(RUN / "final_audit.json")
    assert final["status"] == "passed"
    final["runtime_amendment_verified"] = True
    write(RUN / "final_audit.json", final)
    write(RUN / "completion_audit.json", {
        "status": "passed",
        "final_integrity": final,
        "trained_assessment": trained,
        "note_unit_results_sha256": sha(RUN / "note_unit_results.json"),
        "runtime_amendment_verified": True,
    })
    write(RUN / "completion_progress.json", {"stage": "complete"})
    print("ALL_ASSESSMENTS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
