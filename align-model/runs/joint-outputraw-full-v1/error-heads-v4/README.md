# Error heads v4

This run is the gold-isolated 358-clip validation of a direct-operation plus
learned-probability hybrid. It keeps the completed frozen transcriber, joint
decoder, and v2 error head unchanged.

- Direct candidates come from linked pitch substitutions, acoustically
  supported unlinked events with stable neighboring mappings, and short
  resynchronized DELETE runs.
- Reliability includes exact local n-best operation/span posterior evidence,
  path margin/entropy, acoustic and neighbor confidence, replay state, and
  substitution-versus-delete/insert ambiguity.
- The first deterministic hybrid reached 0.2059 F1. Because it missed the
  material gate while the operation-core oracle remained 0.4724, a small
  validation-only linear selector was fitted on CPU and blended at weight 0.75.

The selected max-F1 and false-controlled policies coincide:

- four-type P/R/F1: **0.2102 / 0.2447 / 0.2262**
- bootstrap 95% F1 CI: **0.2041–0.2504**
- wrong/missed/extra/rhythm F1: **0.1912 / 0.1326 / 0.2445 / 0.0695**
- range-only F1: **0.2789**
- including-repetition historical F1: **0.3237**
- effective false labels: **1,225**, or **3.42/clip** and **4.75/minute**

The recorded validation promotion gate passes, including a 0.0332 max-F1 gain
over v3 and 214.5 fewer false labels than the v3 balanced policy. No production
promotion was performed because the selector was calibrated on this validation
set and the 4,022-row lockbox remains sealed.

Key files:

- `config.json`: experiment/data/resource contract
- `decode_config.json`: deterministic gates, selector coefficients, blend, and
  max/balanced policies
- `calibration.json`: direct-only, learned-only, deterministic-hybrid, and
  selector-hybrid comparisons
- `feature_freeze_manifest.json`: inference-only path-evidence freeze
- `freeze_manifest.json`: final prediction freeze
- `report.json`: official metrics, confidence intervals, decompositions, and
  source/repeat/mapping-correctness breakdowns
- `integrity.json` and `verification.json`: two-process and hash verification
