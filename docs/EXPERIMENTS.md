# Experiment identities and result slots

1. **Original SpeechOcean reference-text ridge**: frozen encoder/phone features; historical penalty/scaling procedure. Deployed as `speechocean-stella-v2`. Original private results apply to this model. Keep its hashes in `speech/experiments/stella-api-v2.json` as provenance until a local export explicitly replaces that config. The original small ridge heads and metadata are bundled under models/speechocean.
2. **seven-families-v1**: completed matched fresh fits, fixed feature combinations, corpus-balanced training, included-corpus development selection. Aggregate results: `results/seven-families-v1.json` (63 rows). Do not call the SpeechOcean row the original deployed model.
3. **Independent seven-family evaluation**: completed matched-cohort aggregate export, all 305 recordings and 54 comparison rows, in `results/external-matched-full-2026-10-18.json`. Statistics were supplied by the evaluator; the repository does not contain private recording-level inputs.
4. **Full representation sensitivity check**: completed `ablation-mac-v1`, 315 candidates and 945 comparisons; recorded .01 promotion rule. See `results/ablation-mac-v1/PROTOCOL.md`, `decisions.json`, and `configs/selected-constructs.json`. All-corpus development selection differs from v1. No automatic deployment or private-data fitting.

The seven-family experiment is not a sequential fine-tuning process. Its hybrid families are fresh ridge fits on balanced corpus mixtures. An earlier anchored Korean/SpeechOcean experiment exists in historical code and must not be substituted for the later matched experiment.

## Public-development sample construction

- SpeechOcean: official train speakers ranked by SHA256(`validation-v1:42:` + speaker); first 25 to development. Official test excluded.
- Korean: duplicate-linked recorder groups assigned via SHA256(`korean-prn-v1:42:` + group), first 8 hex digits modulo 100: <60 train, <80 development, otherwise reserved. Score-blind round-robin samples 4000/1000 before <=20-second and complete-reference filtering.
- AESOP: Japanese L2 speakers sorted by SHA256(`aesop-split-v1-42:` + speaker), 108/36/36 train/development/reserved. Only 6_01, full English reading. Section/rater means are endpoint-mapped.

## Result schemas

Independent table schema: construct, method/model version, reference type (pairwise / other-rater mean / two-rater mean / three-rater mean), N, speaker count, Spearman and CI, QWK and CI, exclusions and clipping notes. Preserve both complete-case and available-case output, labeled clearly. Add paired differences to support comparative claims; marginal intervals are not difference intervals.

Full ablation schema: protocol hash, family, construct, feature blocks, alpha, selection-set identity, evaluation corpus, N, Spearman, MAE, paired uncertainty, eligibility/selection outcome. Selection thresholds are recorded in the completed ablation protocol.
