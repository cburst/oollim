# E2. Documentary audit of the public resources

Version: v1.4. Branch: `essential/dataset-audit`. Base: `cburst/oollim` at `31318b5`.

## What changes

The module adds `contrib/E2/AUDIT.md`. For each corpus, dictionary, pretrained model and source archive used by the release, it records the origin and pinned revision, the license at that revision, the checksum and how it was verified, the split counts against the counts the code enforces, the score scale and its mapping, and the checks against leakage between splits. No code, model or result changes.

## Main observations

- Every pinned weight file (XEUS, PhoneticXeus, XLS-R) has a manifest sha256 equal to the one published by Hugging Face for that revision.
- The SpeechOcean762 split reproduces exactly from the public metadata: 2,000 training and 500 development utterances, 125 speakers, and no speaker shared with the official test partition.
- Twenty-two SpeechOcean762 training references contain words outside the pinned CMUdict entries. `reproduce.extract` stops on such a row, while the historical runner keeps it with incomplete-reference phone features.
- The public J-AESOP rating file on Zenodo uses speaker IDs `JF###` and `JM###` for 183 Japanese speakers. The preparation script expects 180 IDs with the prefix `J_`. The checksum of the rating file inside the corpus archive would document which version was used.
- None of the three encoders declares the learner corpora used here as training data. The released PhoneticXeus configuration lists SpeechOcean762, L2-ARCTIC and the Speech Accent Archive among its development sets. Details are in `upstream-models.md`.

## Files

- `contrib/E2/AUDIT.md`: the audit of corpora, dictionary, models and source archives.
- `contrib/E2/upstream-models.md`: declared training and development data of the three encoders, and their license history.
- `contrib/E2/checksums.txt`: full sha256 of the downloaded weights, checked against the Hugging Face LFS records.

## Dependencies

None.

## How to verify

Each row of `AUDIT.md` names its source and its status, which is `Checked`, `Measured on 10 November 2026` or `Owner`. No `Measure` row remains, since all three were run. The coverage counts need only `resource/scores.json`, the pinned `cmudict.dict` and `pronunciation.text_reference`.

## Results

See `AUDIT.md`. All three Measure items were run on 10 November 2026 and are
recorded there: the audio archive, the CMUdict coverage counts, and the two
reproduction commands. The Owner items remain pending, because they need
archives only the project author holds.

## Changelog

- v1: first audit, read on 10 November 2026.
- v1.1: the three Measure items run and recorded.
- v1.2: eleven corrections to the audit rows and the metadata counts, and the
  grapheme-to-phoneme follow-up shortened to a short paragraph that names what
  a separate module covers instead of specifying it here.
- v1.3: the full sha256 of the J-AESOP rating file, and the follow-up line no
  longer names a branch.
- v1.4: the follow-up paragraph states that the protocol of the separate
  module was fixed before its evaluation, and the reproduction-path row states
  its prediction without pointing at the history that holds it.
