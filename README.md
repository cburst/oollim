# oollim

Expert L2 English speech evaluation on a gaming PC.

Frozen **XEUS**, **XLS-R**, and **PhoneticXeus** representations, reference-phone diagnostics, and supervised **ridge regression** for accuracy, fluency, and prosody.

**Release candidate, 2026-10-19.** The seven-family evaluation, 315-configuration representation ablation, and paired external comparisons are complete. The paired analysis uses 200,000 bootstrap replicates with a second-seed stability check. The [arXiv paper with all appendices](paper/oollim-arxiv.pdf), [Interspeech paper](paper/oollim-interspeech.pdf), and [supplement](paper/oollim-interspeech-supplementary.pdf) are included with LaTeX sources. Private recording-level data and encoder weights are not included.

The neural encoders are frozen. The scoring coefficients **are trained**. Each construct can use its own representation combination and training family.

| Construct | Fixed representation in completed family study |
|---|---|
| Accuracy | XEUS + PhoneticXeus + 17 reference-phone diagnostics |
| Fluency | XEUS + XLS-R |
| Prosody | XEUS + XLS-R |

The seven corpus families are `speechocean`, `korean`, `aesop`, `koreanocean`, `aesopocean`, `koreanaesop`, and `aesopkoreanocean`. Korean fluency and prosody use the **same source label**. Mapping label endpoints to 0–10 does not establish equivalent rating rubrics.

## Local evaluation

Use the [installer and dashboard guide](docs/INSTALL.md) for Apple Silicon macOS and AMD64 Windows. The default installation creates a virtual environment, downloads pinned encoders, and includes all seven ridge families. SpeechOcean is the default; use `--family korean`, for example, to select another. No API key is needed. Platform validation is recorded in [VALIDATION.md](docs/VALIDATION.md).

## Reproduction tests

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,evaluation]'
python -m pytest -q
python -m reproduce.tables
```

The fast tests and table generation need **no corpus, encoder weights, API credential, GPU, or paid service**. `results/seven-families-v1.json` contains all 63 completed aggregate development comparisons. See [`results/seven-families-v1.md`](results/seven-families-v1.md) for readable tables.

## Reproduce the experiments

Follow [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for acquisition, corpus preparation, extraction, training, and local evaluation. The new portable entry points are:

```sh
python -m reproduce.prepare --help
python -m reproduce.extract --help
python -m reproduce.train --help
```

`reproduce/` is the portable release layer. `speech/pronunciation/` contains the historical encoder, pooling, inference and scoring implementations. `speech/scripts/` preserves the relevant research runners and their dependency closure; those historical scripts retain their experiment-specific resource layout and are not the recommended fresh-install entry points.

**Exact reproduction status:** the portable numerical solver is checked against direct normal equations and the historical solver, with synthetic integration tests. The complete raw-corpus → encoder → seven-family rerun has not been performed in a clean environment for this release. It is computationally expensive and must not be confused with the already completed historical experiment. Pinned source and model manifests are included to make this verification possible. See [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Independent evaluation

Evaluation is local through `oollim.py`. The release contains no HTTP server, API client or token setup. Offline commands in `evaluation/run_statistics.py` and `reproduce/external_tables.py` regenerate the compact paper table and full supplementary comparisons. Functions in `evaluation/statistics.py` preserve the Spearman/QWK and paired-bootstrap calculations used for the independent analysis. Private ratings remain on the evaluator’s machine.

## Reproducibility boundaries

- No private recordings, row-level ratings, credentials, or dataset archives are included.
- Small ridge heads for all seven families are included at the author’s direction. Pretrained encoders are downloaded separately; derived-weight permissions remain under review; see [`docs/RIGHTS.md`](docs/RIGHTS.md).
- No private-data calibration or training is performed.
- Development data can guide model choices. Reserved final partitions remain excluded.
- Our speaker-disjoint split does not resolve possible foundation-model pretraining exposure.
- Whole-recording inference currently supports 0.25–120 seconds, up to 400 MiB encoded upload. It is not a real-time claim.
- The original deployed SpeechOcean checkpoint and the matched fresh SpeechOcean refit are different versions.

## Paper and publication

```sh
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The single-file LaTeX manuscript uses standard packages and an inline bibliography, suitable for an arXiv source submission after submission metadata and permissions are finalized. GitHub owner is `cburst`; the repository name is `oollim`; visibility remains to be confirmed; no public repository URL or DOI is invented.

## License

Original code and documentation are **source-available** under [the Custom Source-Available License (Non-Production Use Only)](LICENSE.md), copyright 2026 R ROSE. Production use requires a separate license. This is not a permissive open-source license. Third-party model and dataset terms remain separate; a license from the author does not grant rights to those resources.

## Completed sensitivity analysis

All 315 representation/family/construct candidates are documented in `results/ablation-mac-v1/`. The original .01 promotion threshold is retained. `configs/selected-constructs.json` identifies the unchanged overall heads: koreanocean accuracy and aesopkoreanocean fluency/prosody. See [reproduction commands](docs/REPRODUCE.md).
