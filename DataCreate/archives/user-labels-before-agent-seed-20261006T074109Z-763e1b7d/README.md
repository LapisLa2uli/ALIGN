# User-label archive

All prior active user-label files are in `user_labels_before.zip`, byte-for-byte. Entry paths are relative to DataCreate. `agent_labels_snapshot.zip` stores copied sources. `manifest.json` records all paths and SHA-256 hashes, including the missing-agent sample.

To restore, close annotation editors, back up any newer manual edits, and extract `user_labels_before.zip` to DataCreate with its relative paths. A manifest row with user_before_sha256=null means no original user file existed.

Seeded files retain agent provenance and are pending human review. This operation makes no claim about model accuracy or false-negative rates.
