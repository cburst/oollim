# X4. Evaluation on the official SpeechOcean762 test partition

Version: v1 (draft). Branch: `extra/04-speechocean-test`. Base: `cburst/oollim` at `31318b5`.

## What changes

The module evaluates the released `speechocean` scoring heads, and fixed refits of them, once on the official SpeechOcean762 test partition (2,500 utterances, 125 speakers). It reports Pearson correlation, Spearman correlation and mean absolute error for sentence-level accuracy, fluency and prosody, with speaker bootstrap intervals. The protocol is recorded before any test number is computed, and the test partition informs no choice. No released file changes.

## Systems

- S1: the released `speechocean` heads for the five selection rules, as distributed.
- S2: the same block sets and penalties refitted on the full official training partition (2,500 utterances), with the release fitting code.
- S3: one ridge per block combination (XEUS, XLS-R, PhoneticXeus, phone diagnostics), with the penalty chosen on the 500 development utterances.
- S4: the repository's fusion network on the S1 block sets, five seeds.

## Reference coverage

Seventeen test references contain words outside the pinned CMUdict entries. The primary analysis keeps all 2,500 utterances and uses the incomplete-reference phone features that `phone_evidence` produces for them. A secondary analysis uses the 2,483 covered utterances, matching what the application accepts.

## Upstream exposure

The released PhoneticXeus training configuration lists a SpeechOcean762 development set whose transcripts come from a split named after the official test partition (see `contrib/E2/upstream-models.md`). For each construct, results of systems that use PhoneticXeus features are therefore reported next to the best system without PhoneticXeus features.

## Files

- `contrib/X4/prepare.py`: the manifest for the full train partition and for the
  official test partition. The train manifest is byte-identical to the one
  `reproduce.prepare speechocean` writes.
- `contrib/X4/extract.py`: resumable extraction. Calls the same
  `UnifiedScorer.extract`, accepts the test partition, records reference
  coverage per row, and saves the PhoneticXeus logits and the pooled mean and
  standard deviation of every encoder layer in the same pass.
- `contrib/X4/evaluate.py`: S1 to S4, the metrics, the speaker bootstrap
  intervals, the tables and the saved predictions.
- `contrib/X4/sample.py`: the fixed validation subset used before the full run.
- `contrib/X4/compare.py`: bitwise comparison of two feature caches, used to
  show that the cost options change no value.
- `contrib/X4/PROTOCOL.md`: recorded before evaluation, with the deviations and
  the clarifications appended later.
- `contrib/X4/results/`

## What the module does not reproduce

The original neural training configuration is not present in the repository:
`experiments/unified-neural-v1.json`, which `speech/scripts/train_unified.py`
reads, is absent. The protocol fixes the learning rate, the weight decay and
the epoch count, and this module fixes the batch size at 64 with the
`FusionNetwork` defaults. Those four values are the module's own, they are on
the list of questions for the author, and no S4 figure should be read as
reproducing the released neural comparison.

## Dependencies

E2 for the coverage rule. No GPU and no private data.

## How to verify

Rerun the scripts on the public corpus. Feature extraction uses the release
`UnifiedScorer.extract`, and the tables are rebuilt from the saved predictions
without extracting again.

Three checks run on their own:

- the train manifest must have the sha256 the release manifest has,
- the pooled output of the last saved layer of each encoder must equal the
  release feature vector, on every utterance, not on a sample,
- `compare.py` must report the cache produced with `--verify-once --trim`
  bitwise identical to the cache produced without them.

## Results

Pending.

## Changelog

- v1: protocol and scripts.
