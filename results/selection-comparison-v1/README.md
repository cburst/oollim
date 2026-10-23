# Completed selection-rule comparison

All 21 combinations completed the same 305 recordings (6,405 assessments). The original historical SpeechOcean evaluation remains a separate comparison. The public release contains aggregate statistics and public-trained weights only.

| Condition | Public development criterion | Normalized penalty grid |
|---|---|---|
| `spearman` | Each construct's Spearman; ties lower MAE, then larger penalty | .001, .01, .1, 1, 10, 100 |
| `damae` | Accuracy MAE; ties larger penalty | Same |
| `historical_grid_damae` | Accuracy MAE; ties larger penalty | .00005, .0005, .005, .05, .5, 5 |

The historical grid is the original summed-loss grid divided by the original 2,000 SpeechOcean training utterances. This fixed conversion applies to every corpus-balanced mean-loss family. Historical-grid DAMAE does not recreate every original implementation convention. Fluency and prosody use the accuracy-selected penalty for their shared feature combination. Fitting, standardization, and selection use public training/development partitions only.

## Main-table choice

Report every construct-specific leader within .001 of the unrounded maximum for Spearman or QWK across all 21 combinations. Both metrics are shown for each selected configuration. These are descriptive external-evaluation maxima, not prospectively selected winners or statistically established ties. The full matrix retains matched comparisons for all three rules. The application defaults and the public-development construct choices remain unchanged.

Historical-grid koreanocean improves two-rater fluency QWK from .585 to .744 and prosody QWK from .546 to .675, with rank correlations .838 and .840. The .744 fluency point estimate slightly exceeds human consensus .739; no paired superiority claim is made for that sensitivity contrast. All alternative results remain disclosed. Selecting a new preferred configuration from these external outcomes would be a post-evaluation choice.

## Contents

- `external-all-conditions.csv`: 144 aggregate rows, including both reference sizes, human baselines, Azure, all 21 combinations, historical SpeechOcean, clipping counts, and marginal 95% intervals.
- `external-matched.json`: 54 main-comparison aggregate rows; matched SpeechOcean replaces the historical deployment.
- `tables/`: complete matched-Spearman comparison rows, regenerated from aggregates.
- `paired-200k-seed42.json`, `paired-200k-seed2026.json`: recomputed 90-contrast matched-Spearman paired analysis. Adjusted interval decisions agree for 89/90 contrasts. Matched SpeechOcean prosody correlation versus Azure is borderline (adjusted lower endpoints +.000027 and -.002153).
- `training-protocol.json`: original recorded grid and public manifest hashes.
- `audit.json`: completion, receipt, and point-statistic verification.

All 144 Spearman/QWK point estimates were independently checked against the private recording-level export. Marginal intervals retain the supplied 2,000-draw, seed-42 bootstrap. Paired matched-Spearman intervals use 200,000 draws with a second seed; the 90-contrast correction does not cover selection among all sensitivity conditions. URL grouping assumes one recording per speaker. QWK clips to 0–100 and rounds half-up; Spearman uses raw scores. No calibration is fitted.

## Reproduction

From the repository root, regenerate the main table without private data:

```bash
python -m reproduce.leader_tables --output output/external-tables
```

Run each public-data condition with its own output directory:

```bash
python -m reproduce.train --speechocean CACHE_S/manifest.json --korean CACHE_K/manifest.json --aesop CACHE_A/manifest.json --condition spearman --output output/spearman
```

Repeat with `--condition damae` and `--condition historical_grid_damae`. Use `--extractor` with the pinned extractor metadata for deployable heads; see the main reproduction instructions for cache acquisition/extraction. The exact evaluated weights and their checksums are in `models/selection-comparison-v1/registry.json`.

An authorized holder of the private data can verify the supplied archive locally:

```bash
python -m reproduce.selection_tables selection_evaluation_results.zip --output output/selection-audit
python evaluation/paired_review.py selection_evaluation_results.zip --member stella305_all_selections.csv --prefix spearman__ --bootstrap 200000 --seed 42 --output output/paired42.json
```

Repeat the paired command with seed 2026. Private recordings, filenames, IDs, URLs, row-level scores, and request receipts are not distributed. The public aggregate release reproduces tables but does not let readers independently recalculate private-data statistics without access to that collection.
