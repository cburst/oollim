# Install and run oollim

Initial targets: **Apple Silicon macOS (native ARM64)** and **Windows 11 AMD64**. Use native 64-bit Python **3.11**. CUDA, WSL, an API account, and a separate FFmpeg installation are not required. The installer supplies a CPU runtime and FFmpeg. Internet access is required for installation and pinned model acquisition; evaluation runs locally.

Download/extract this source repository into a writable folder. Install the matching Python 3.11 distribution from [python.org](https://www.python.org/downloads/). In a terminal opened in the repository:

macOS:
```sh
python3.11 install.py
python3.11 oollim.py
```

Windows PowerShell:
```powershell
py -3.11 install.py
py -3.11 oollim.py
```

The default installs into `.venv`; no activation is necessary. `install.py --current` explicitly installs into the invoking Python instead. `--plan` prints the plan, and `--skip-models` defers encoder downloads. Installation logs are under `work/install`. Rerun after an interrupted download; verified assets are reused. Do not run the installer with administrator/sudo privileges.

Allow approximately 10 GB for dependencies and encoder files, plus temporary download space. Memory requirements increase sharply with recording duration. A supported CPU architecture does not guarantee enough RAM for a two-minute recording. Start with a short recording, close memory-heavy applications, and inspect the per-recording log for memory failures. CPU processing is not real time.

## Evaluate

No arguments opens the interactive prompt and Rich dashboard. Commands below use `python` to mean the interpreter/launcher used above.

```sh
python oollim.py --audio "recording.m4a" --transcript "reference.txt" --output scores.csv
python oollim.py --folder "audio" --transcript "common-reference.txt" --output scores.csv
python oollim.py --folder "audio" --recursive --output scores.csv
```

Without `--transcript`, every audio/video needs a matching UTF-8 sidecar: `example.mp3` → `example.txt`. With `--transcript`, that single text is used for every recording. Reference text is required. Video containers are decoded using their first audio stream. Whole recordings must be 0.25–120 seconds and no larger than 400 MiB. Longer files are rejected, never silently shortened.

The default bundle is the original deployed **speechocean** model: XEUS + PhoneticXeus + IPA for accuracy, XEUS + XLS-R for fluency and prosody. `--model path/to/family.json` selects a compatible authorized bundle. All seven research families are included at the author’s direction; choose one with `--family korean` (or `aesop`, `koreanocean`, `aesopocean`, `koreanaesop`, `aesopkoreanocean`). Their source-data and derived-weight permission status is recorded in RIGHTS.md.

Scores use the trained 0–10 scale. Raw linear predictions can exceed its endpoints; the application does not silently clip or recalibrate them. Record the model version in any evaluation.

## Resume and diagnostics

Completed records are saved after each file, in the CSV and its `.state.json` companion. Keep both. Resume checks audio content, reference text and the model manifest. Changed inputs are rescored. Rerun the same command to reuse completed records; add `--retry-failed` to retry failures. Close the CSV in Excel before running. One writer per output is allowed; different model runs should use different output paths.

`Ctrl+C` stops the active worker and its decoder; completed records remain saved. A forced OS shutdown can leave a `.lock`; verify that no evaluation process remains before removing it. Temporary decoded audio is removed on ordinary completion or cancellation. Your original files are untouched. Outputs and logs contain filenames and scores; keep them private when evaluating private recordings.

```sh
python oollim.py --doctor
python oollim.py --download-models
python oollim.py --folder audio --retry-failed --no-live --threads 4 --timeout 3600
```

`--doctor` verifies imports, bundled FFmpeg, source and model checksums. `--runtime-only` omits weight checks. Installation success is not an end-to-end inference test; see the validation record for actual platform coverage.

## Native Windows acceptance test

After extracting the release ZIP, open PowerShell in its `oollim` folder. Substitute a short valid recording and its UTF-8 reference text:

```powershell
py -3.11 install.py
if ($LASTEXITCODE -ne 0) { throw "Installation failed; inspect work/install" }
py -3.11 oollim.py --verify-install --audio "C:\path\recording.m4a" --transcript "C:\path\reference.txt"
```

This runs the doctor, scores one whole recording, and reruns it to check resume. Reports/logs are written under `work/acceptance`. Share the compact JSON report first; no audio or reference upload is needed. Native Windows acceptance remains pending until this test succeeds on Windows.
