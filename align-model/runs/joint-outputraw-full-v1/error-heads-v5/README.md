# Error heads v5

V5 is the leakage-free estimate of the direct-operation plus learned hybrid.
It does not use the 358 validation targets for selector fitting, blend
selection, thresholds, clustering, or padding.

Protocol:

- 3,634 train rows / 1,846 leakage groups fit the linear operation-core
  selector.
- 910 train rows / 475 disjoint leakage groups calibrate the five predeclared
  learned/direct/hybrid candidates and schema policies.
- 358 validation rows / 194 groups are absent from both train partitions.
- Selector, decode configuration, row IDs, and artifact hashes were frozen
  before the validation target database was opened.
- Validation targets were opened exactly once in a separate scoring process.

One-shot validation:

- learned-only four-type F1: **0.1914**
- direct-only four-type F1: **0.2005**
- selected hybrid P/R/F1: **0.1654 / 0.2845 / 0.2092**
- hybrid bootstrap 95% F1 CI: **0.1910–0.2287**
- wrong/missed/extra/rhythm F1: **0.1916 / 0.1272 / 0.1880 / 0.0786**
- range-only/type-only F1: **0.2666 / 0.4387**
- including-repetition F1: **0.2925**
- max-F1 false labels: **5.34/clip**
- high-precision policy: **0.1662 F1**, **2.18 false labels/clip**

Exploratory v4 scored 0.2262 after fitting on these same validation targets, so
its apparent optimism is 0.0170 F1. Honest v5 improves 0.0178 over its
train-calibrated learned-only baseline, narrowly missing the predeclared 0.02
material gate. The high-precision policy also regresses below learned-only.
V5 is therefore not promoted.

Artifacts:

- `protocol_manifest.json`: explicit fit/calibration/validation ordinals,
  groups, hashes, and disjointness declaration
- `selector.json`: train-fit operation-core selector
- `calibration.json`: train-calibration candidate metrics
- `decode_config.json`: predeclared frozen policies and multiplicity
- `freeze_manifest.json`: validation inference artifacts produced without gold
- `VALIDATION_OPENED.json`: immutable one-shot opening record
- `report.json`: official independent validation metrics
- `integrity.json` and `verification.json`: protocol and artifact checks

The 4,022-row lockbox remained metadata-only and production was unchanged.
