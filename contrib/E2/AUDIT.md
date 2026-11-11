# E2. Documentary audit of the public resources

Base: `cburst/oollim` at `31318b5`. Read on 10 November 2026. No model was retrained.

Status of each item:

- **Checked**: confirmed by reading the code and the primary source, with hashes recomputed where a file was downloaded.
- **Measure**: needs the full source files on the execution machine. The command is given.
- **Owner**: needs an archive that only the project author holds.

## SpeechOcean762

| Item | Finding | Status |
|---|---|---|
| Origin | Audio: OpenSLR 101 archive `speechocean762.tar.gz` from the ELDA mirror, 520,810,923 bytes. Metadata: `jimbozhang/speechocean762` at commit `613968e3`, 15 files. The manifest holds 16 SpeechOcean762 entries, these 15 and the OpenSLR archive. | Checked |
| License | CC BY 4.0, stated on the OpenSLR 101 page. | Checked |
| Checksum | OpenSLR publishes the size (520M) but no checksum. The sha256 values in `speech/resources-manifest.json` are the project's own record. All 15 metadata files pinned in the manifest were downloaded at the pinned commit and their sha256 values match it: `README.md`, `resource/scores.json`, `resource/scores-detail.json`, `resource/lexicon.txt`, `resource/text-phone`, and `utt2spk`, `text`, `wav.scp`, `spk2age` and `spk2gender` for each of `train` and `test`. | Checked |
| Audio archive | Downloaded from the OpenSLR 101 ELDA mirror and verified: sha256 `876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5` and 520,810,923 bytes, both equal to `speech/resources-manifest.json`. The unpacked archive holds 5,245 WAV files in 250 speaker directories, all 16 kHz, mono, PCM_16, with none unreadable. Of these, 5,000 are referenced by `train/wav.scp` and `test/wav.scp` (2,500 each), and every referenced file is present. The remaining 245 files are referenced by neither partition and have no entry in `resource/scores.json`. They sit in 36 `SPEAKER96xx` directories, which also hold 720 utterances that the partitions do use, 420 in train and 300 in test, so the directory is not the criterion and the 245 are identified by absence from both `wav.scp` files rather than by their identifiers. Durations run from 1.590 s to 20.408 s, mean 4.103 s, median 3.584 s, p95 7.680 s, 21,520.9 s in total (5.98 h). | Measured on 10 November 2026 |
| Splits | The official train partition has 2,500 utterances from 125 speakers, 20 per speaker. `reproduce/prepare.py` ranks speakers by sha256 of `validation-v1:42:<speaker>` and assigns the first 25 to development. This gives 2,000 training and 500 development utterances, the counts in the paper. The official test partition (2,500 utterances, 125 speakers) is not used. | Checked |
| Leakage | No speaker and no utterance ID is shared between the official train and test partitions. Training and development are speaker-disjoint by construction. `reproduce/train.py` also rejects shared speakers or audio hashes between training and development at run time. | Checked |
| Scale | Sentence-level `accuracy`, `fluency` and `prosodic` are integers from 0 to 10. According to the corpus README, they are the average or median of five experts. Training ranges are 3 to 10, 1 to 10 and 1 to 10. The mapping to the common scale is the identity. | Checked |
| Reference coverage | Twenty-two training references (21 in the training split, 1 in development, 16 speakers) contain words absent from the pinned CMUdict first entries, for example `henny`, `steeven` and `layla's`. In the official test partition, 17 references (14 speakers) are affected. Every count was recomputed by running `pronunciation.text_reference.text_to_phones` with the pinned dictionary over `resource/scores.json`, and all six match: 22 and 16 in the official training partition, of which 21 fall in the training split and 1 in development, and 17 and 14 in the official test partition. The 16 absent training words are `balt`, `deeps`, `dora's`, `hadi`, `hazarded`, `henny`, `jayme's`, `jumpped`, `layla's`, `liya`, `mandy's`, `nacy's`, `steeven`, `stoped`, `viewings` and `wolfs`. | Measured on 10 November 2026 |
| Reproduction path | `reproduce.prepare speechocean` keeps all 2,500 rows, and `reproduce.extract` raises `Incomplete reference coverage` on such a row. The historical runner `speech/scripts/run_fusion_baselines.py` keeps those rows, and `phone_evidence` gives them its incomplete-reference values (`reference_complete` 0, edit rates 0, CTC terms 0). Both commands were run. `prepare` wrote 2,500 rows, 2,000 in the training split and 500 in development, and placed `000480019` at position 105. `extract` then printed progress through `104/2500`, cached 104 feature records, and raised `ValueError: Incomplete reference coverage` on the next row, which is `000480019`, whose absent word is `balt`. Reading the code, and before either command was run, this file predicted that extraction would stop at `000480019`. The measurement confirms it. | Measured on 10 November 2026 |

Measurement note. `reproduce.extract` needs the pinned ESPnet as an importable
package. `docs/REPRODUCE.md` installs it with pip from the upstream commit. Here it
was instead placed on `PYTHONPATH` from `speech/resources/source/wanchichen-espnet-5b52d57`,
the copy the installer extracts and verifies file by file against
`.oollim-source.json`. The imported code is the same pinned commit either way, and
the substitution is recorded because the documented command was not the one run.

Measurement note. The audio archive row was measured with an independent reader that opens each
WAV header directly, not with `speech/scripts/acquire.py`. The checksum and byte count are the
ones the manifest records, so the verification is the same and only the tool differs. The count of
annotated, partitioned utterances is 5,000, and 5,245 is the number of files the archive ships.

## Korean Learners' L2 Speech Corpus (English), AIHub 71463

| Item | Finding | Status |
|---|---|---|
| Origin | AIHub dataset 71463, Training PRN archives `TS_PRN_en_NA_NA.zip` (audio) and `TL_PRN_en_NA_NA.zip` (labels, inside `korean-train-labels.zip`). The official Validation archive is not used. | Checked |
| License | AIHub usage policy, with access by application. Corpus redistribution is excluded. `docs/RIGHTS.md` records the author's permission to release the derived weights. | Checked against `docs/RIGHTS.md`. The terms themselves are not verified here. |
| Checksum | No upstream checksum. `reproduce/prepare.py` writes a local receipt with the sha256 of each supplied archive. | Owner |
| Splits | Speaker groups are recorder IDs joined when their decoded audio is identical. A deterministic hash of the group gives a 60/20/20 split, and the last 20 percent is sealed. Duration anomalies are quarantined (at most 915). A score-blind selection takes 4,000 training and 1,000 validation candidates. Preparation then drops recordings longer than 20 s or with incomplete CMUdict coverage, and it raises unless 2,699 and 636 rows remain. These are the counts in the paper. | Checked by reading. The counts need the archive (Owner). |
| Exclusions | 1,301 of 4,000 training candidates and 364 of 1,000 validation candidates are excluded. The reasons per row are written to `exclusions.json`. | Owner |
| Scale | `articulationScore` and `prosodyScore` from 1 to 5, mapped with 2.5(y − 1). Accuracy uses the articulation score. Fluency and prosody both use the prosody score. | Checked |
| Leakage | Grouping by recorder ID plus identical decoded audio. The same person under two recorder IDs with no identical audio cannot be detected from the available identifiers. | Checked |

## J-AESOP

| Item | Finding | Status |
|---|---|---|
| Origin | Corpus archive `CorpusData-main` with `Audio/` and `Documents/Rating6_01.csv`, under written permission. Corpus paper: Yazawa, Konishi and Kondo, Acoustical Science and Technology 46(5):553 to 563, 2025, doi:10.1250/ast.e25.06. Public ratings: Zenodo, doi:10.5281/zenodo.10907820. | Checked |
| License | Audio under an individual agreement, as described in `docs/RIGHTS.md`. Zenodo ratings under CC BY 4.0. | Checked |
| Checksum | Zenodo `Rating6_01.csv`: md5 `9801871bc74efceead5d61382be55e46` and 274,801 bytes, both as the Zenodo API publishes them, and sha256 `a4b3249840d76371484a2f6e4ddc34693965ab80fdf240e259d3f50fe48e4009`, computed on 10 November 2026 from the file downloaded at that record, whose md5 and size were checked against the published values first. Zenodo itself publishes md5 only. The corpus archive has no published checksum. | Checked (Zenodo). Owner (archive). |
| Speaker IDs | The Zenodo rating file lists 183 Japanese speakers (IDs `JF001` to `JF115` and `JM001` to `JM068`) and 25 English speakers. `reproduce/prepare.py` expects exactly 180 speakers whose IDs start with `J_`, and audio named `<ID>_6_01.wav`. The rating file in the corpus archive therefore differs from the Zenodo release in ID format, since the Zenodo file has no `J_` identifier at all, and the speaker count also differs, 183 against 180. | Checked (Zenodo). Owner (archive file and its checksum). |
| Splits | Speakers ranked by sha256 of `aesop-split-v1-42:<speaker>`. The first 108 are training, the next 36 development and the last 36 reserved. These are the counts in the paper. | Checked by reading |
| Scale | In the Zenodo file, `Segment`, `Fluency` and `Prosody` are integers from 1 to 10, with 48 ratings per speaker (16 raters × 3 sections). The target is the mean over all 48, mapped with 10(y − 1)/9. Accuracy uses `Segment`. | Checked |
| Leakage | One recording per speaker, so the splits are speaker-disjoint. All speakers read the same passage. | Checked |

## CMUdict

| Item | Finding | Status |
|---|---|---|
| Origin | `cmusphinx/cmudict` at commit `74790861`, file `cmudict.dict`. | Checked |
| License | BSD-style license of Carnegie Mellon University (1993 to 2015), from `LICENSE` at the pinned commit. | Checked |
| Checksum | sha256 `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22` recomputed and equal to the manifest. The manifest records no size. | Checked |
| Use | Only the first, unnumbered entry of each word is used (`cmudict-first-entry-text-v1`). Alternative pronunciations are ignored. Both reading passages (North Wind and the Sun, Please Call Stella) are fully covered. | Checked |

## Pretrained models and code

| Resource | Origin and revision | License at that revision | Checksum | Status |
|---|---|---|---|---|
| XEUS weights | `espnet/xeus` at `b6512612`, `model/xeus_checkpoint_new.pth`, 2,306,233,340 bytes | CC BY-NC-SA 4.0 (model card) | Manifest sha256 equals the sha256 published by the Hugging Face LFS record | Checked |
| PhoneticXeus weights and code | `changelinglab/PhoneticXeus` at `ed64f14a`, `model.safetensors`, 2,300,089,432 bytes, plus 39 code, configuration and documentation files pinned one by one | CC BY-NC-SA 4.0 (model card, marked as an update). The GitHub source at `1be0aecb` carries the same license | Weight sha256 equals the Hugging Face LFS record. The five code and configuration files downloaded here (`attention.py`, `e_branchformer.py`, `layer_norm.py`, `config.json`, `xeus_config.yaml`) match the manifest | Checked |
| XLS-R 300M | `facebook/wav2vec2-xls-r-300m` at `1a640f32`, `pytorch_model.bin`, 1,269,737,156 bytes | Apache 2.0 (model card) | Manifest sha256 equals the Hugging Face LFS record | Checked |
| ESPnet source for XEUS | `wanchichen/espnet` at `5b52d57b`, GitHub generated tarball | Apache 2.0 (`LICENSE` at the commit) | The manifest sha256 is the project's own record. GitHub does not guarantee byte-stable generated archives, so the commit is the stable identifier. The installer also stores per-file hashes in `.oollim-source.json`. | Checked |

The manifest also pins a GitHub tarball of PhoneticXeus at `1be0aecb`. Inference imports the Hugging Face files at `ed64f14a`, and the tarball serves as an acquisition record.

Upstream training-data exposure of the pretrained models is covered in `upstream-models.md`.

## Commands for the Measure items

```bash
python -m reproduce.prepare speechocean --source /path/to/speechocean762 --output work/so-audio
python -m reproduce.extract --manifest work/so-audio/manifest.json --output work/so-features
```

The coverage count needs no audio. Running `pronunciation.text_reference.text_to_phones` with the pinned dictionary over `resource/scores.json` reproduces the 22 and 17 references listed above.

All three Measure items have now been run on this machine. The audio archive was
measured with the independent reader described above, the coverage count was
reproduced, and the two commands were executed with the pinned ESPnet on
`PYTHONPATH`.

## Possible follow-up

References with words outside the pinned CMUdict entries are rejected by the
application and receive incomplete phone features in training. A follow-up module for
words outside the dictionary is offered separately. Its protocol, fixed before
the evaluation, names the grapheme-to-phoneme models, the measures of
pronunciation accuracy, and an application option that is off by default.
