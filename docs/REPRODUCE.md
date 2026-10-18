# Reproduction procedure

Run commands from the repository root, in Linux/WSL or macOS. Python 3.11 is the research target. FFmpeg and a C/C++ runtime supporting the chosen PyTorch build are system prerequisites. Set BLAS thread counts explicitly if sharing a machine. None of these commands starts a paid API run.

## 1. Obtain resources and install encoder dependencies

Read `RIGHTS.md` before obtaining datasets or distributing checkpoints. The acquisition script downloads only public resources listed in its manifest; it does not accept credentials or download the private evaluation.

```sh
python -m pip install -e '.[encoders]'
python speech/scripts/acquire.py
```

`speech/resources-manifest.json` pins revisions and SHA-256 checksums. The observed research environment is recorded in `configs/requirements-research-observed.txt`; it is an environment record, not a claim that every platform can install every pinned version. CPU and GPU PyTorch builds must match their torchaudio versions. Use the source commit of the XEUS ESPnet fork, not an arbitrary current ESPnet release:

```sh
python -m pip install --no-deps 'espnet @ git+https://github.com/wanchichen/espnet.git@5b52d57b4f872ff7babded35316a79642e9e6c12'
```

The adapters load local pinned weights and validate source hashes. They do not fetch or execute unpinned remote code during extraction. Third-party source archives in the manifest are acquisition records; do not blindly extract them over this repository. The ESPnet installation above supplies its importable package. PhoneticXeus support files are individually downloaded under `speech/resources/phoneticxeus/`. CMUdict is downloaded under `speech/resources/text-reference-v1/`.

## 2. Acquire and prepare corpora separately

- SpeechOcean762: obtain the official archive from OpenSLR 101 and unpack it into a directory containing `train/`, `test/`, `resource/` and the audio tree.
- Korean: obtain the approved AIHub 71463 Training PRN archives. Put `korean-train-labels.zip` (containing `TL_PRN_en_NA_NA.zip`) and `TS_PRN_en_NA_NA.zip` together. The audit verifies every audio member, groups decoded duplicates, quarantines duration anomalies, then selects score-blind candidates. This can take hours and reads substantial data.
- J-AESOP: obtain permission and the source `CorpusData-main` ZIP containing `Audio/` and `Documents/Rating6_01.csv`. Do not redistribute it with the code.

Replace `/path/to/...` below with your local authorized copies:

```sh
python -m reproduce.prepare speechocean --source /path/to/speechocean762 --output work/so-audio
python -m reproduce.prepare korean --source /path/to/korean-archives --output work/ko-audio
python -m reproduce.prepare aesop --source /path/to/corpus.zip --output work/ae-audio
```

The output manifests contain local audio paths, reference text, recording/speaker IDs, split, audio checksum, and targets ordered accuracy/fluency/prosody. **Do not publish these generated manifests**, especially for restricted datasets. They are ignored when placed under `work/`.

Expected cohorts: SpeechOcean 2000/500; Korean 2699/636; J-AESOP 108/36 train/development recordings. Korean preparation checks these exact final counts. Differences require investigation and a new protocol, not silent cohort substitution. No final-test features are requested by these entry points.

## 3. Extract frozen features

```sh
python -m reproduce.extract --manifest work/so-audio/manifest.json --output work/so-features
python -m reproduce.extract --manifest work/ko-audio/manifest.json --output work/ko-features
python -m reproduce.extract --manifest work/ae-audio/manifest.json --output work/ae-features
```

This portable extractor processes whole recordings sequentially on CPU with two Torch threads by default. It loads one encoder at a time and resumes only cache entries whose audio, reference, metadata and feature checksums match. Expect extraction to dominate runtime. The release wrapper retains pooled vectors; the historical `speech/scripts/extract_frames.py` and frame-cache modules support the separate sequence/hierarchy experiments. Re-extraction need not overwrite an existing research frame cache.

Pinned extraction metadata is in `configs/extractor.json`. It carries model names, revisions, native configurations, code checksums and text/phone provenance. Its inherited regression fields are historical provenance only; the release extractor does not load or use its old scoring coefficients.

## 4. Fit the completed matched protocol

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m reproduce.train \
  --speechocean work/so-features/manifest.json \
  --korean work/ko-features/manifest.json \
  --aesop work/ae-features/manifest.json \
  --extractor configs/extractor.json \
  --output work/seven-families --mode fixed --selection included
```

This fits 21 selected construct heads. The arrays retain all three output columns for compatibility, but each deployed construct uses its own selected head. The runner rejects private/final rows, overlapping speakers or exact audio across train/development, reordered features, nonfinite values, and changed input checksums. Corpus identity independence beyond available identifiers is not inferred.

Compare `work/seven-families/comparison.json` with `results/seven-families-v1.json`. Small numerical differences across BLAS/Torch builds are possible; unexplained cohort, preprocessing or large metric differences must be resolved before claiming reproduction. The original historical runner is `speech/scripts/run_seven_families.py`; it documents the exact completed run's layout and includes-cache selection rule.

The historical deployed SpeechOcean model has different earlier tuning. Recreate its 15 reference-text variants from the same feature manifest with:

```sh
OPENBLAS_NUM_THREADS=2 python -m reproduce.original --manifest work/so-features/manifest.json --output work/original-speechocean
```

This preserves the original StandardScaler, sum-squared-loss alpha grid, accuracy-MAE selection, and grid-order tie rule. Compare with `results/original-speechocean-v1.json`. The API-selected combinations were X+P+IPA for accuracy and X+R for fluency/prosody. Package a refit with a new family manifest and checksums before selecting it using `--model`. Refits receive a new model version; do not claim bitwise identity until prediction parity is checked. The old original checkpoint is not silently replaced or renamed.

## 5. Completed representation sensitivity check

The completed `ablation-mac-v1` sweep contains 315 candidates and 945 development comparisons. Keep the recorded **0.01** promotion threshold; the proposed 0.005 alternative was not adopted. All three development corpora inform selection, including corpora absent from a family's training set; these are not untouched cross-corpus tests.

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python -m reproduce.train \
  --speechocean work/so-features/manifest.json \
  --korean work/ko-features/manifest.json \
  --aesop work/ae-features/manifest.json \
  --output work/ablation --mode all --selection all
python -m reproduce.select_ablation --results work/ablation --output work/ablation-selection
```

The archived run omitted `--extractor`; add it only if intentionally creating a separately versioned run. Full protocol, aggregate candidate/penalty results, completed promotion decisions, and the exact training-code snapshot are in `results/ablation-mac-v1/`. The original run used public feature caches on an M1 Mac with two numerical threads and completed in approximately 25 minutes. Raw corpus extraction is much slower. Checkpoint files and per-recording development predictions are generated locally; they are not distributed as candidate bundles.

The selection script compares shortlisted combinations against the same-protocol fixed-combination baseline, applies all per-corpus gates, and runs the paired speaker bootstrap. It rejects incomplete sweeps. A numerical leaderboard winner is not automatically a promoted model. No command changes shipped model bundles.

Final overall choices: koreanocean accuracy (XEUS + PhoneticXeus + IPA), aesopkoreanocean fluency and prosody (XEUS + XLS-R). `configs/selected-constructs.json` pins their existing checkpoint hashes. These are unchanged heads; previously generated predictions remain reusable. The CLI still accepts one family at a time: evaluate koreanocean for accuracy and aesopkoreanocean for the other two scores, or use the manifest when integrating construct-specific selection. AESOP-only fluency/prosody improvements are development candidates, not replacements for the models in the external paper table.

## 6. Local evaluation

Use `oollim.py --family speechocean --audio recording.wav --transcript reference.txt` (or another family). There is no server to configure. See INSTALL.md for folder processing and resume. Historical configuration filenames containing `api` are retained only where a research runner needs their provenance; no HTTP server or API client is shipped.

## 7. Reproduce the paper tables offline

Render the supplied, rounded aggregate export without accessing private recordings or ratings:

```sh
python -m reproduce.external_tables --output work/paper-tables
```

This produces the compact nine-row table in Markdown/HTML, exact LaTeX body rows, and all 54 supplementary comparison rows in CSV. The main table uses two-rater consensus for models and other-two-rater consensus for humans, with both Spearman and QWK. This is rendering supplied aggregates, not recomputing their statistics.

An evaluator who holds recording-level scores can recompute both tables locally:

```sh
python -m evaluation.run_statistics assembled_scores.csv --speaker-column speaker_id --output work/private-statistics --bootstrap 2000 --seed 42
```

Required columns: `h1-AccuracyScore` through `h3-ProsodyScore`, `m-AccuracyScore`/`m-FluencyScore`/`m-ProsodyScore`, and `<family>-<Construct>Score100` for all seven families, all on 0–100. Original `s-<Construct>Score` columns may supply SpeechOcean values on 0–10 and are multiplied by ten. Supply verified speaker identities; a URL is suitable only when each recording belongs to a unique speaker. Empty rows are ignored. Missing/incomplete results are rejected for final compact rendering. No network, model training, calibration, or audio inference occurs. Keep inputs and generated row-level files out of the repository.
