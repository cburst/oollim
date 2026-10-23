# Release verification

## Checked locally

Current local test run: **22 passed** (3.17 seconds), fresh Python 3.11 ARM64 environment. Source syntax compilation and a targeted credential/private-path scan also passed.

- Weighted ridge coefficients agree with independently solved weighted normal equations in both primal/dual shapes, including a constant feature.
- Duplicating all rows while halving their weights preserves the solution.
- Portable family solver matches the archived completed-run solver on identical synthetic inputs.
- A synthetic three-corpus run creates all 21 heads and 63 development result rows; saved heads reproduce saved predictions.
- The original SpeechOcean sum-loss solver agrees with direct normal equations.
- Train/development speaker overlap, final-test rows, and changed feature checksums are rejected.
- Offline evaluation tests cover identifier matching, QWK clipping/raw-rank behavior, missing data, and human/system consensus rows. API transport has been removed from this release.
- All 63 public development aggregate rows are retained; all 15 original SpeechOcean representation variants are retained separately.
- Pinned text-reference and phone-feature implementation hashes match the bundled historical source.
- The manuscript compiles with pdfLaTeX and its rendered pages were visually checked. The built-in compiler was unavailable on this host, so the installed TeX Live compiler was used.

## Not yet verified for this release

- Native Windows installation and inference. The owner will run the supplied `oollim.py --verify-install` check later. WSL does not count as native Windows validation.
- A full raw-corpus extraction/training rerun for this release.
- Bitwise equality of new encoder extraction across CPU/GPU, BLAS, Torch, and OS versions. Use pinned versions and compare predictions/metrics, not only file hashes.
- Full representation ablation and the completed aggregate private-family export are now included; private recording-level statistics were not independently recomputed.
- GitHub Actions: workflow included, but no hosted run exists until publication.

## Delegated review

A bounded Luna review independently extracted the completed-run methodology. A bounded Sol review inspected the new wrappers. The extraction cache now includes wrapper and implementation hashes. The review's shared Korean-target concern is explicitly documented as a source-label limitation; it is not silently “fixed” by inventing an independent fluency score. Source archive receipts are verified by the reused audit function. Multi-output checkpoints identify the selected construct in training metadata.

No running research API was stopped or reconfigured for this release, and no private evaluation audio was opened.

## Installer and local operation (2026-10-18)

- Fresh isolated `.venv` installation passed on Apple Silicon macOS, including pinned Torch/torchaudio, dependencies, pinned ESPnet source and the runtime doctor.
- `pip check` reports no broken dependencies.
- Bundled FFmpeg decoded WAV and MP4 audio with spaces and Korean characters in paths. Tests reject malformed media and recordings above 120 seconds.
- All seven bundled family manifests pass file checksums and extraction-provenance compatibility checks.
- Tests cover content-sensitive resume, CSV regeneration, malformed bundles, corrupted completed partial downloads, and archive traversal rejection.
- `.gitattributes` preserves LF source bytes on Windows because checkpoint provenance checks source hashes.
- Native Windows acceptance is prepared but pending; GitHub Actions has not run.

The distribution has two user entry points: `install.py` and `oollim.py`. The short public recording acceptance test passed as recorded below.

### End-to-end acceptance: passed

Native Apple Silicon macOS, Python 3.11.10, Torch 2.11.0; fresh isolated environment. All 49 pinned resources were downloaded and SHA-256 verified. No existing research dependency environment was used for inference.

The complete SpeechOcean762 training recording `011970107` (8.21 seconds; “SO LISA WENT ON TO PLAYGROUND”) was processed on CPU with four threads. No cropping, segmentation or score calibration was used.

| Check | Result | Elapsed |
|---|---|---:|
| Runtime, FFmpeg and resource doctor | Pass | 8.10 s |
| Decode → three encoders/IPA → ridge scores → CSV | Pass | 42.13 s |
| Identical rerun reuses saved result | Pass; zero new inference | 8.30 s |

Default `speechocean-stella-v2` scores: accuracy 7.083750, fluency 6.380180, prosody 6.306598. This is an operational smoke test, not a new model-quality experiment or a long-recording capacity benchmark. All 20 distinct head files across seven families also loaded and produced finite synthetic-feature predictions; full audio inference was tested with the default family.

The 22-test suite passed in the fresh environment. Native Windows testing remains pending at the owner’s request; the repository is a Mac-tested release candidate, not yet a Windows-verified release.

## Reproduction update, 2026-10-18

26 tests passed, including aggregate completeness, selected checkpoint hashes, exact compact-table parity with the manuscript, and incomplete-sweep rejection. The portable selection command reran all paired bootstrap checks from the completed public caches and reproduced the 21 decisions. The offline statistics CLI passed a synthetic 20-speaker end-to-end check. No private inputs were used. Clean ZIP validation uses the tested Mac Python environment; this does not replace the pending native Windows test.

## Paired external analysis, 2026-10-19

The evaluator supplied the complete 305-row score archive. Recording-level inputs remain outside the repository and were used only for evaluation. All 48 recomputed two-rater system estimates match the supplied export within 1e-10. The generic `evaluation/paired_review.py` command reads a local evaluation ZIP and produces aggregate results: 20,000 paired bootstrap draws, seed 42, 90 contrasts with marginal and Bonferroni-adjusted percentile intervals. URL-based grouping follows the evaluator's one-recording-per-speaker convention. No inference, training, calibration, or model selection was performed on these inputs. Public release includes only aggregate summaries and analysis code. The actual supplied marginal-table settings are 2,000 draws, seed 42.

The paired analysis was repeated with 200,000 draws using seeds 42 and 2026. All 90 adjusted interval-exclusion decisions agree between seeds and with the earlier run; maximum endpoint discrepancy is 0.00366834. Both aggregate exports are retained. The supplement now includes all 315 representation macro correlations and the 21 promotion decisions, with all 945 per-corpus results in ablation-corpus-results.csv. Two AESOP-only alternatives passed within-family gates; deployed external scores and overall construct selections were unchanged.

## Three-condition comparison revision

All 21 family/selection combinations completed all 305 external recordings. The release audit verifies all 6,405 completed receipts against the augmented data and independently recomputes the 144 Spearman/QWK aggregate rows. All 63 bundled construct heads match their evaluated checksums. Twenty-six targeted solver, condition-selection, save/restore, aggregate-release, manuscript-table, and local-application tests passed. Main-table paired differences were recomputed at 200,000 draws for two seeds; 89/90 adjusted interval decisions agree, with matched SpeechOcean prosody versus Azure borderline. PDFs were compiled and key table pages visually checked. Private evaluation records remain outside the repository. Application defaults and running API services were not changed by this publication revision.

## Construct-leader table revision (2026-10-23)

All 27 ridge, reproduction-release, and local-app tests pass. The generated table contains seven Spearman leaders/near-ties and three QWK leaders, plus the human and Azure baselines for each construct. The .001 threshold uses unrounded estimates. PDFs compile without overfull-box or undefined-reference warnings; the Interspeech main text remains within four pages, with references continuing onto page five. Models and application defaults are unchanged.

## Version 0.3: 21 scoring combinations

All 49 tests pass, covering all 21 checksum-verified bundles, GUI family/rule routing, selection-dependent resume identity, and preservation of original bundles. A real 8.21-second public sample was decoded and encoded once and scored with all 63 construct heads. Native macOS Cocoa GUI scoring and PDF export passed using the new matched-Spearman default; a separate CLI run passed with koreanocean/DAMAE. The new Platypus installer is rebuilt but requires owner signing/notarization. These tests reuse installed dependencies and encoder assets; they do not represent a fresh Mac installation. Windows EC2 validation is recorded separately in `validation-0.3.0.json`.

The final Windows Server 2022 AMD64 installer passed its native wizard, dependency/model setup, combined-selector audio evaluation, PDF export, and Open PDF action. The exact tested executable was downloaded and SHA-256 verified on the Mac. Generated Python bytecode caches are excluded from packaging because startup can modify them before payload verification. Windows 11 remains unverified.
