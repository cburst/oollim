# E1. Release PhoneticXeus attention maps during inference

Base: `cburst/oollim` at `31318b5`.

## What changes

`speech/pronunciation/inference_memory.py` registers a global forward hook that clears the attention probability tensor that ESPnet attention layers keep in `self.attn` after each forward pass. The hook matched only the module path `espnet.nets.pytorch_backend.transformer.attention`, which is the path XEUS uses. PhoneticXeus loads a vendored copy of the same classes from `pxeus.espnet_import.attention`. Its `forward_attention` also stores `self.attn`, so the hook skipped it, and every PhoneticXeus layer kept its attention map until the forward pass over the whole recording ended.

The change adds `pxeus.espnet_import.attention` to the accepted module paths. The class name check and the tensor check are unchanged.

The match on `__module__` is an exact comparison against a fixed list, which is a known limitation. A copy of these classes reached by a different import path, for instance through `trust_remote_code` under another package name, would not be matched and would keep its maps. Attention arithmetic, layer outputs, pooling and scoring heads are untouched.

Two different records hold hashes, and only one of them changes. The checkpoint provenance records the hashes of six files, `encoders.py`, `scoring.py`, `phonetic.py`, `audio.py`, `phone_features.py` and `text_reference.py`, and none of them is modified, so every bundled head still loads. `docs/source-snapshot.json`, which records the source of the local release, has one entry updated. That makes `speech/pronunciation/inference_memory.py` the only entry whose `release_sha256` differs from its `source_sha256`, which says that the released copy of this one file diverges from the upstream source it was taken from. Whether to record that, or to leave the snapshot as it is, is the author's choice, and the snapshot update is a separate commit for that reason.

## Why it matters

For a recording with T encoder frames (T = (samples − 400) // 320 + 1 at 16 kHz), each PhoneticXeus layer keeps 8 heads × T² float32 values, that is 32·T² bytes, and the 19 layers keep 608·T² bytes at the end of the pass. At 120 s (T = 5999) that is 20.38 GiB of attention maps that inference never reads.

## Measured effect

CPU, float32, one process per run under a 10 GiB memory cap, on Linux with Python 3.11.15, torch 2.11.0+cu130 and transformers 5.19.0.

- Retention: before the change, the bytes kept after layer i equal (i + 1) × 32·T² to the byte at 15, 30, 45 and 60 s. After it, all 372 layer observations of the three encoders keep zero bytes.
- Peak growth of PhoneticXeus inference: the fitted quadratic coefficient is 716 bytes per frame² before the change (15 to 60 s), against 704 predicted, and 131 after it (15 to 120 s), against 128 predicted. The 15 to 120 s range for the corrected code was chosen after the measurements. Over 15 to 60 s that fit gives 47. A separate analysis made after the measurements largely explains why: below about 54 s the peak is set by a term linear in the duration, from the stages before the first encoder block, rather than by the attention transient. It is not a complete account, because the same analysis fits 192 rather than 128 for XEUS over the narrow range, which remains unexplained.
- Peak resident memory at 120 s after the change: 7.62 GiB, measured. Before the change, 120 s does not fit on the 15 GiB measuring host, and the comparable extrapolation from 15 to 60 s is 27.1 GiB, give or take about 0.9 GiB. The capacity test in the manuscript records 22.52 GiB of resident memory with 5.51 GiB of swap at 120 s, which is consistent with the extrapolation.
- Features and scores: the four pooled blocks and all 105 head outputs are bitwise identical before and after the change at 15, 30, 45 and 60 s. Two runs of the unchanged code are also bitwise identical.
- XEUS and XLS-R, used as controls, are unchanged by the fix. Six of their eight before-and-after comparisons agree within 0.4 per cent. The two that do not, XEUS at 60 s by 5.9 per cent and XLS-R at 15 s by 4.4 per cent, are run-to-run variation rather than an effect of the change, which cannot reach either encoder because the unchanged code already matched the module path they use.

The predicted coefficients and the retention formula were stated before the measurements were run. Every measured figure above is regenerated from the raw outputs by `contrib/E1/recompute.py` on the evidence branch named at the end of this card, into `contrib/E1/results/recomputed.json`.

## Tests

`tests/test_inference_memory.py` adds six tests. Five need no downloaded assets and pin the fix itself.

- The hook clears `attn` in inference mode and keeps it in training mode, on both module paths.
- The vendored path is written out literally, so removing it from the accepted list fails a test.
- A stand-in class on the module path of the PhoneticXeus encoder layer, which owns an `attn` submodule, and an attention class from another package, are left alone. The test builds a stand-in rather than the real layer, which would need the downloaded weights.
- A module on an accepted path whose `attn` is a submodule rather than a tensor is left alone.

The sixth runs only when the PhoneticXeus source files have been downloaded. It builds the vendored `MultiHeadedAttention` class and checks that the hook clears its `attn`.

## What changes for users

- Scores do not change. Features and scores were bitwise identical at every duration where both versions run, 15 to 60 s.
- `docs/source-snapshot.json` records the new `release_sha256` of `speech/pronunciation/inference_memory.py`. The other 46 recorded hashes are unchanged.
- Feature caches written before the change raise `Changed extraction inputs` and must be rebuilt, because `reproduce/extract.py` records the hash of every file in `speech/pronunciation/`. Rebuilding produces the same features, bit for bit.

## How to verify

1. `python -m pytest -q tests/test_inference_memory.py`
2. The full suite gives the same results with and without the change. On the measuring host, with the desktop dependencies absent and `tests/test_gui.py` excluded, it gives 3 failed and 48 passed before the change and 3 failed and 54 passed after it, with the same three failures by name. Those two counts come from logs on the evidence branch, not from this one. Two come from the optional `nltk` and `PySide6` dependencies, and one from label notation in `paper/main.tex`.
3. The memory measurements and the identity check are reproduced with the scripts and commands on the evidence branch.

## Evidence

The measurement harness, the raw outputs, the recomputation script, the measurement protocol and the full logs are on the `evidence/E1` branch of the fork this pull request comes from, under `contrib/E1/`.
