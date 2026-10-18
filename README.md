# 🎙️ oollim

**Expert L2 English speech evaluation on a gaming PC.**

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/macOS-Apple_Silicon-000000?logo=apple&logoColor=white)
![Windows AMD64](https://img.shields.io/badge/Windows-AMD64-0078D4)
![Local inference](https://img.shields.io/badge/Inference-local_CPU-success)
[![License](https://img.shields.io/badge/License-source--available-orange)](LICENSE.md)

## 🚀 Install first

Download and extract the repository into a writable folder, then install native **64-bit Python 3.11** from [python.org](https://www.python.org/downloads/). Open a terminal in the extracted `oollim` folder.

### 🍎 Apple Silicon Mac

```sh
python3.11 install.py
python3.11 oollim.py
```

### 🪟 Windows 11 · AMD64

```powershell
py -3.11 install.py
py -3.11 oollim.py
```

🧰 The installer creates a `.venv`, installs dependencies and FFmpeg, and downloads pinned speech encoders with progress indicators. No environment activation, API key, CUDA, or WSL is required. Launching `oollim.py` opens the interactive prompts and Rich progress dashboard.

💾 Allow approximately **10 GB** for dependencies and encoders, plus temporary download space. Internet access is needed for installation; evaluation runs locally. Start with a short recording: memory use rises sharply with audio duration, and CPU inference is not real time.

⚙️ Use `install.py --plan` to preview installation, `--skip-models` to defer model downloads, or `--current` to install into the invoking Python instead of a virtual environment. Do not use sudo or administrator privileges.

📖 See the [installation and troubleshooting guide](docs/INSTALL.md) for full instructions. Apple Silicon has passed the recorded smoke test; **native Windows acceptance testing remains pending**. See [platform validation](docs/VALIDATION.md).

## 🎧 Score a recording or folder

A reference transcript is required. On macOS:

```sh
# 🎵 One recording and its transcript
python3.11 oollim.py --audio "recording.m4a" --transcript "reference.txt" --output scores.csv

# 📁 A folder sharing one reference text
python3.11 oollim.py --folder "audio" --transcript "reference.txt" --output scores.csv

# 📝 A folder with a matching .txt transcript beside each recording
python3.11 oollim.py --folder "audio" --recursive --output scores.csv

# 🧪 Choose a different training family
python3.11 oollim.py --folder "audio" --transcript "reference.txt" --family korean --output korean.csv
```

🪟 On Windows, replace `python3.11` with `py -3.11` in these commands.

- 🎬 FFmpeg decodes supported audio and video containers automatically.
- ⏱️ Whole recordings must be **0.25–120 seconds** and no larger than **400 MiB**.
- 📊 Output includes accuracy, fluency, and prosody scores on the trained **0–10 scale**. Raw ridge predictions can exceed those endpoints; no silent clipping or calibration is applied.
- 🔄 Keep the CSV and its `.state.json` companion to resume. Rerun the same command; add `--retry-failed` to retry failures.
- 🩺 Run `oollim.py --doctor` with the same Python launcher to check dependencies and model assets.

## 🧠 What produces the scores?

Frozen **XLS-R**, **XEUS**, and **PhoneticXeus** representations combine with reference-phone diagnostics and supervised **ridge regression**. The encoders remain fixed; the scoring coefficients are trained. Each construct can use its own representation combination and training family.

| 🎯 Construct | 🧩 Representation in the completed family study |
|---|---|
| Accuracy | XEUS + PhoneticXeus + 17 reference-phone diagnostics |
| Fluency | XEUS + XLS-R |
| Prosody | XEUS + XLS-R |

👪 Seven training families are included: `speechocean` (default), `korean`, `aesop`, `koreanocean`, `aesopocean`, `koreanaesop`, and `aesopkoreanocean`.

## 🔬 Reproduce the work

### ⚡ Fast tests and published tables

These checks require **no corpus, encoder weights, API credential, GPU, or paid service**. From the repository folder on macOS:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test,evaluation]'
.venv/bin/python -m pytest -q
.venv/bin/python -m reproduce.tables
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test,evaluation]'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m reproduce.tables
```

📊 [Completed development tables](results/seven-families-v1.md) report all 63 aggregate comparisons; the underlying data are in `results/seven-families-v1.json`.

### 🏋️ Corpus preparation, extraction, and training

Follow the [experiment reproduction guide](docs/REPRODUCE.md) for acquisition, corpus preparation, feature extraction, training, and evaluation. Using your environment’s Python:

```sh
python -m reproduce.prepare --help
python -m reproduce.extract --help
python -m reproduce.train --help
```

🗂️ `reproduce/` provides portable entry points, `speech/pronunciation/` contains encoder and scoring implementations, and `models/` holds the trained ridge bundles. `speech/scripts/` preserves historical research runners with their experiment-specific layouts.

✅ The seven-family evaluation, **315-configuration representation ablation**, and paired external comparisons are complete. The paired analysis uses **200,000 bootstrap replicates** and a second-seed stability check. The ablation retains the **.01 promotion threshold**; `configs/selected-constructs.json` records the unchanged overall selections: koreanocean accuracy and aesopkoreanocean fluency/prosody. Full ablation results are in `results/ablation-mac-v1/`.

🔎 The portable solver is checked against direct normal equations and the historical solver, with synthetic integration tests. A complete clean-environment raw-corpus → encoder → seven-family rerun has **not** been performed for this release. Pinned source and model manifests support that verification; see [validation details](docs/VALIDATION.md).

## 🧑‍🏫 Independent evaluation

Evaluation runs locally through `oollim.py`. Offline commands in `evaluation/run_statistics.py` and `reproduce/external_tables.py` regenerate the compact paper table and supplementary comparisons. `evaluation/statistics.py` preserves the Spearman, QWK, and paired-bootstrap calculations. Private ratings remain on the evaluator’s machine.

## 📌 Scope and provenance

- 🔒 No private recordings, row-level ratings, credentials, or dataset archives are included; private evaluation data are not used for training or calibration.
- 📦 Small ridge heads for all seven families are included at the author’s direction. Encoders download separately; derived-weight permissions remain under review. See [rights and restrictions](docs/RIGHTS.md).
- 🧭 Development partitions guide model choices; reserved final partitions remain excluded.
- 🔍 Speaker-disjoint splits do not resolve possible pretrained-model exposure to evaluation corpora.
- 🏷️ The default SpeechOcean bundle is the original API-evaluated ridge. The matched fresh SpeechOcean refit uses the standardized family-comparison training procedure and has different coefficients; results belong to their respective versions.

## 📄 Paper and supplementary materials

- 📚 [arXiv-format paper with all appendices](paper/oollim-arxiv.pdf) · [LaTeX source](paper/oollim-arxiv.tex)
- 🎤 [Interspeech-format paper](paper/oollim-interspeech.pdf) · [LaTeX source](paper/oollim-interspeech.tex)
- 📎 [Interspeech supplement](paper/oollim-interspeech-supplementary.pdf) · [LaTeX source](paper/oollim-interspeech-supplementary.tex)

To build the complete arXiv-format edition with a LaTeX installation:

```sh
cd paper
pdflatex -interaction=nonstopmode -halt-on-error oollim-arxiv.tex
pdflatex -interaction=nonstopmode -halt-on-error oollim-arxiv.tex
```

📝 These are manuscript drafts supplied with the reproduction package; inclusion does not imply publication or acceptance.

## ⚖️ License

Original code and documentation are **source-available** under the [Custom Source-Available License (Non-Production Use Only)](LICENSE.md), copyright 2026 R ROSE. Production use requires a separate license. Third-party model and dataset terms remain separate; the author’s license does not grant rights to those resources.
