# ALIGN data audit v2

This release is read-only with respect to every source corpus. It audited
26,294 bundle-shaped directories (26,200 synthetic and 94 real) and admitted
23,233 synthetic bundles with validated exact note lineage.

## Release artifacts

- `audit_report.json`: machine-readable counts, severities, examples,
  distributions, cache status, and audits of prior frozen manifests.
- `split.json`: source-group-isolated manifest with 14,718 train, 1,871
  validation, and 6,644 locked test bundles.
- `validated_targets.jsonl.gz`: one hash-locked written-pitch target record per
  manifest row, including corrected raw-MIDI rendered events and exact
  clean/performed lineage.
- `validated_targets.sqlite`: indexed, compressed copies of the same records
  for random-access training through `alignmodel.validated_targets`.
- `heldout_protocol.json`: immutable evaluation rules and artifact hashes.
- `bundle_index.json`: per-bundle provenance, validity, pitch-space, duration,
  class, source, and file hashes.

## Material findings and repairs

- All three prior non-smoke frozen manifests use source-concentrated prefixes;
  the current `outputRaw_sf_10k` first 1,000/200/200 subsets are 100% Mozart.
- The `outputRaw_sf_10k` split places both Mozart and Weber in train,
  validation, and test. The combined historical split also crosses raw source
  groups between train and validation.
- The rendered portion of 25,552 note maps disagreed with raw MIDI event
  count/time/pitch or pitch-space conventions. The target cache deterministically
  rebuilds those rendered events from raw MIDI without changing source maps.
- 23,233 maps pass the corrected exact-lineage checks; 2,967 synthetic bundles
  remain excluded. Major original faults include 2,319 performed-map/XML
  mismatches, 2,036 clean-map/XML mismatches, 640 missing exact maps, and 8
  missing rendered maps.
- 1,640 bundles contain 2,320 labels outside the WAV duration. Those labels,
  plus 47 repetition labels inconsistent with copy lineage, are omitted from
  target sidecars; valid note-lineage supervision is retained.
- Of 2,278 Basic Pitch/PESTO cache files inspected, 1,741 Basic Pitch and 336
  PESTO caches match WAV hashes and pitch policy; 100 of each have stale pitch
  policy.
- The focused hash-matched acoustic audit measured 752 intonation events:
  none matched their labels within 20 cents and 98.0% measured within 20 cents
  of zero. All 42,213 unverified intonation labels in admitted rows are
  therefore excluded from `usable_labels`.

## Locked split policy

Raw snippets are grouped by source score across corpora; generated melodies
are grouped by clean-score SHA-256. Weber is locked to test, score `001` to
validation, and Mozart to train. Remaining generated groups use seeded
80/10/10 assignment. Rows are ordered round-robin by corpus/source so common
first-N consumers no longer select an all-Mozart prefix.

The test split must not be used for thresholding, model selection, or
qualitative iteration. The F1 >= 0.80 value remains a target, not a result
established by this audit.
