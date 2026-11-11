# X4 protocol, recorded before any test number is computed

Date recorded: 10 November 2026.
Base commit: `31318b5833c2d7b6ab2b86a4ad4a1b1b4e1dbd5c` of `cburst/oollim`.

Erratum, 11 November 2026, recorded with the results: the full base commit is `31318b5fb91e50b69dc4283f7ac506b58cbaa890`, the upstream `main`. The 40-character value above was written in error and resolves to no object. The base was always the upstream `main` at `31318b5`, which is the merge base of this branch.

This protocol is fixed before any script in this module has produced a number
from the official SpeechOcean762 test partition. Everything that follows is
fixed as of the date above.

## Objective

Evaluate, once, on the official SpeechOcean762 test partition, the
sentence-level accuracy, fluency and prosodic scores. The partition holds 2,500
utterances from 125 speakers with integer ratings from 0 to 10. The test
partition informs no choice of any kind. In the test partition fluency and
prosodic reach 0, while in the training partition the minimum of both is 1.

## Systems, all specified before evaluation

S1. The published heads of the `speechocean` family under the five selection
rules, exactly as distributed:

- `models/selection-comparison-v1/spearman`
- `models/selection-comparison-v1/damae`
- `models/selection-comparison-v1/historical_grid_damae`
- `models/banded-3ratio-v1/banded_3ratio_spearman`
- `models/banded-3ratio-v1/banded_3ratio_accuracy_mae`

S2. A refit of each S1 head on the 2,500 utterances of the official training
partition, keeping the same blocks, the same penalty and the same
`weighted_fit` procedure, meaning standardization computed on the training rows
only and a per-block gain of the square root of 2048 divided by the block
width. Nothing is selected again.

S3. The 15 non-empty combinations of the four blocks XEUS, XLS-R, PhoneticXeus
and IPA, crossed with the three constructs. Fitting uses the 2,000 utterances
of the existing project split, and the penalty is chosen on the 500
development utterances by the project rule, which prefers Spearman
correlation, then mean absolute error, then the larger penalty. This system is
descriptive.

S4. The `FusionNetwork` of `speech/pronunciation/fusion.py` over the S1 block
sets, trained by the procedure in `speech/scripts/train_unified.py` with AdamW
and the epoch chosen by accuracy mean absolute error on the development split.
Five seeds, 0 through 4. Learning rate 1e-3, weight decay 1e-2 and 300 epochs.
These three values are fixed here because the original training configuration
is not present in the repository. Results are reported as mean and standard
deviation across seeds.

## Reference coverage rule, fixed before evaluation

T-all is the primary analysis and keeps all 2,500 test utterances. The 17
references whose words fall outside the pinned CMUdict first entries receive
the incomplete-reference phone features that `phone_evidence` already
produces.

T-cov is the secondary analysis and keeps the 2,483 covered utterances, which
matches what the application accepts.

Both are reported. S2 uses the 2,500 training utterances under the same
treatment.

## Upstream exposure, declared before evaluation

The released PhoneticXeus configuration lists a development set named
`dev_speechoceanotth`, which reads its transcripts from
`test_speechoceannotth`. The name suggests the official test partition, and
such a set may have informed the choice of checkpoint. Therefore, for each
construct:

- systems that include the PhoneticXeus block and the best system without the
  PhoneticXeus block, which comes from S3, are reported side by side
- every row that includes the PhoneticXeus block is marked as potentially
  optimistic

## Metrics

Pearson correlation is primary. Spearman correlation and mean absolute error
are reported per construct, computed on the raw predictions. A variant of
Pearson correlation is also reported with predictions clipped to the interval
from 0 to 10. Intervals are 95 percent speaker bootstrap intervals with 2,000
resamples and seed 42.

## Scripts

No file of the release is modified.

1. `prepare.py` writes a manifest for the complete training partition and for
   the official test partition, in the format `reproduce.prepare speechocean`
   produces. A separate script is needed because `load_corpus` refuses the test
   partition by design.
2. `extract.py` follows the logic of `reproduce/extract.py` with two changes.
   It accepts the test partition and it does not raise on incomplete reference
   coverage. It records `reference_complete` per row and calls the same
   `UnifiedScorer.extract`. It also saves the PhoneticXeus logits per utterance
   for later use by module X7. Inference runs on CPU with 2 threads per
   process, as the reproduction instructions specify. The work may be split
   into parallel shards as long as the total stays within the memory limit and
   the division is recorded with the results.
3. `evaluate.py` runs S1 through S4, computes the metrics and the bootstrap
   intervals, and writes the tables to `contrib/X4/results/` together with the
   saved predictions, so the tables can be rebuilt without extracting again.

## Amendment of 10 November 2026, per-layer outputs saved in the same pass

Recorded before any extraction has run. The extraction pass of `extract.py`
also saves, for every utterance, the pooled mean and standard deviation of all
layers of the three encoders. The purpose is to avoid repeating a pass that
costs 6 to 10 hours of CPU time.

Sources of the layer outputs:

- XEUS. `speech/pronunciation/encoders.py` already requests
  `return_all_hs=True`, so all 19 layer outputs are already available.
- XLS-R. `Wav2Vec2Model` is already called with `output_hidden_states=True`.
- PhoneticXeus. Per-block forward hooks on the encoder, which only read the
  block output and never modify it.

Rules fixed here:

- The pooled output of the final layer, the one that feeds the heads, must be
  identical to the one the release path produces. A test of this on a sample of
  utterances is recorded with the results.
- Format is one file per encoder, for example `layers_<encoder>.npy`, of shape
  (N, L, 2d) in float32, together with the list of utterance identifiers. This
  is about 0.8 GB for XEUS over 5,000 utterances and about 3 GB in total, kept
  under /home.
- These extra outputs exist only for modules X2 and X5. They enter no analysis
  in X4. No system, metric or table of this module reads them.

## Expected cost

Extraction of the 5,000 utterances takes 6 to 10 hours of CPU time. Ridge
fitting and bootstrap take minutes. S4 takes under one hour. No GPU is used and
no private data is involved.

## Literature figures, for the final text only

These numbers are recorded here so they cannot be used to choose anything in
this module. They appear only in the final write-up, as published reference
points.

- GOPT, fluency 0.753 and prosodic 0.760
- HMamba, fluency 0.848 and prosodic 0.843
- Accuracy, and the figures of Kim et al. 2022, are to be confirmed against
  the primary source at writing time

## Deviations

Any departure from this protocol is to be recorded in this file, in a later
commit, with the reason and the date.

### Recorded on 10 November 2026, before any test number was computed

D1. ESPnet on PYTHONPATH. `docs/REPRODUCE.md` installs the pinned fork with
`pip install --no-deps 'espnet @ git+...@5b52d57b...'`. The scripts instead put
the copy the asset installer extracts and verifies file by file,
`speech/resources/source/wanchichen-espnet-5b52d57`, on the path. The imported
code is the same pinned commit, but the command run is not the documented one.
Already recorded in `contrib/E2/AUDIT.md`.

D2. An extra attention-release hook. `extract.py` installs a global forward
hook that clears `module.attn` on the vendored `pxeus.espnet_import.attention`
classes as well as the ESPnet ones. The released
`speech/pronunciation/inference_memory.py` filters on the ESPnet module path
alone, so the vendored copy retains 19 maps totalling `19*20*32*T^2/2` bytes,
about 6.3 GB on a 20 s utterance. Module E1 established that clearing them
leaves every feature bitwise identical across four durations and 372 layer
observations. No release file is modified, and the hook lives in `extract.py`.
Without it, parallel shards would not fit in memory. `--keep-attention`
reproduces the release behaviour.

Erratum, 11 November 2026, recorded after the evaluation and changing no
result: the retained size given above is wrong. Module E1 measured the
retention after layer i as (i + 1) x 32 T^2 bytes, so at the end of the pass
the 19 vendored layers keep 608 T^2 bytes, which is 606,784,608 bytes, about
0.61 GB, for a 20 s utterance (T = 999), not about 6.3 GB. The figure above
summed the retention over the layers instead of taking its value at the end of
the pass, and it is larger by a factor of exactly 10. Its own expression,
19*20*32*T^2/2, evaluates to 6.07 GB rather than the 6.3 GB written beside it,
so that figure was imprecise as well. The hook changes no feature, as E1
established, so no number in this module depends on it.

The statement that parallel shards would not fit without the hook rested on
that figure and was not measured. The `--keep-attention` flag exists and would
allow the measurement, and nothing in this module or in E1 records it. What was
measured is the opposite configuration, in note C9: two shards with the hook
active held about 3.1 GiB of resident set each.

D3. Verification hashed once per process. `UnifiedScorer.extract` re-reads
7.618 GiB of weights on every call, measured at 16.24 s of a 27.09 s call, and
`model.safetensors` is hashed twice because both `UnifiedScorer.extract` and
`PhoneticXeus.__init__` verify it. `--verify-once` memoizes the hash of each
asset, keyed by device, inode, size and modification time, so a file that
changes under the process is hashed again. This weakens the release guarantee
from once per utterance to once per process plus a stat per call. Over the
20-utterance validation sample the memoization served 1,750 of 1,800 hash
calls, hashing 50 distinct files once, which confirms that patching
`encoders.sha256` and `fusion.sha256` reaches every caller.

D4. `malloc_trim` after each utterance. `UnifiedScorer.extract` loads and
deletes three encoders per call, and PyTorch CPU tensors come from
`posix_memalign`, so the freed weights stay in glibc arenas and resident memory
grows to roughly the sum of the three models. Returning the arenas to the
kernel lowered the peak from 7.365 GiB to 5.764 GiB, which is what decides how
many shards fit in 15 GiB of RAM. It changes no value.

Evidence for D2, D3 and D4, recorded in `results/validation-arms.json`. The
20-utterance validation sample of `sample.py` was extracted twice: arm A with
the release verification behaviour and no trim, arm B with `--verify-once
--trim`. `compare.py` compared the two caches array by array: 20 of 20 rows
bitwise identical over nine array names, the four feature blocks, the three
layer stacks, the logits and the greedy token ids, with a largest absolute
difference of 0.0 everywhere. Arm B took 298.6 s against 609.4 s, a factor of
2.04. The default of `extract.py` repeats the release behaviour, and these
three deviations stand only because the comparison is exact.

D5. Amendment of 10 November 2026: a read-only cross-layer pooled output.
Recorded before the extraction that produces it and before any number from it.
Module X12 needs LM2, the pooled mean and standard deviation over frames of the
mean over layers of the frame sequence. The mean of LM2 follows from the
per-layer pooled means by linearity, but the standard deviation does not: the
variance of a mean over layers is `(1/L^2)` times the sum of every cross-layer
covariance, and only the diagonal of that matrix survives per-layer pooling.
Recovering it later would cost a second pass over the audio, so it is computed
in this one.

It is computed exactly rather than accumulated. The capture already holds every
layer of a forward pass, so the mean over layers of the frame sequence is
formed in full and then pooled by the release `masked_pool`, the same function
and the same epsilon the per-layer pooling uses. There is no sum and sum of
squares, and therefore no catastrophic cancellation to reason about. For
PhoneticXeus, whose block outputs are pooled as they arrive to keep the peak
down, a single running sum of the frame tensors is kept instead, which costs
one tensor of `T` by 1024 in float32 and is likewise exact.

Two index sets are saved per encoder, so X12 can choose without a second pass:

| Encoder | `all` | `core` |
|---|---|---|
| XEUS | 0 to 18, the 19 block outputs | identical to `all`, there being no index to drop |
| XLS-R | 0 to 24, all 25 saved entries | 1 to 24, dropping the feature projection at index 0 |
| PhoneticXeus | 0 to 19, all 20 saved entries | 0 to 18, dropping the `after_norm` output at index 19 |

The XEUS `core` row is written anyway and is identical to its `all` row by
construction, so the file has one shape for every encoder.

Saved as `crosslayer_<encoder>.npy` of shape `(N, 2, 2048)` in float32, row 0
being `all` and row 1 being `core`, beside `layers_<encoder>.npy`. These
outputs exist only for X12 and X2. They enter no analysis in X4, and no system,
metric or table of this module reads them.

Two checks run on every utterance. The identity rule of the amendment of the
per-layer outputs still applies unchanged. In addition, the mean half of the
`all` row must equal the mean over layers of the per-layer pooled means to
float64 tolerance, which is the one quantity derivable two ways, and a
mismatch aborts the run.

### Clarifications of points the protocol left open

C1. Layer counts. The amendment says "all layers" without fixing a count. The
counts were measured: XEUS 19, one tensor per block before the final norm, the
adapter selecting index -1. XLS-R 25, the feature projection at index 0 and
the 24 transformer layers after it. PhoneticXeus 20, its 19 blocks followed by
the
encoder `after_norm` output. The PhoneticXeus heads pool the post-norm
representation, not the last block: `after_norm(block 18)` equals
`PhoneOutput.frames` bitwise while block 18 alone differs by up to 18.74, so
the post-norm representation is the twentieth saved layer and the one the
identity rule applies to.

C2. The identity test covers every utterance. The amendment asks for a test on
a sample. The three comparisons cost nothing beside the forward pass, so they
run on every utterance and a single mismatch aborts the run.

C3. Upstream exposure covers the `ipa` block too. The protocol marks rows that
include the PhoneticXeus block. `phone_evidence` derives the `ipa` block from
the PhoneticXeus logits, so it carries the same exposure. Rows whose blocks
include either are marked, and the paired system without exposure is drawn
from the block sets that contain neither.

C4. Amendment of 10 November 2026: the criterion for "the best system without
the PhoneticXeus block". Recorded before this module produced any number from
the test partition. The protocol does not fix the criterion, and choosing it by
test performance would be a selection on the test partition. The rule is
therefore the one module X6 already uses for its prespecified systems: the
highest development Spearman correlation, ties broken by the lower development
mean absolute error, on the same 500-utterance development split the penalty
selection uses.

X6 averages that Spearman over the SpeechOcean762, Korean and J-AESOP
development sets with equal weights, and reports a development-size-weighted
variant beside it. S3 fits on one corpus and selects on one development set, so
the mean has a single term and the two weightings select the same system. Both
weightings are therefore reported by the single row the tables carry, which
says so, rather than by two rows that would be identical.

The system that is best by test Pearson correlation is reported beside it and
marked as descriptive, because that figure does involve the test partition.

C5. S4 needs two networks, not fifteen. The `scaler_mean`, `scaler_scale` and
`block_gain` of the released heads are bitwise identical across the five
selection rules for a given block set, so the standardized design matrix the
network reads depends on the block set alone, and the released heads use two
block sets. `evaluate.py` asserts the identity rather than assuming it and
raises if a future release breaks it.

C6. The S4 batch size is not fixed by the protocol, and
`experiments/unified-neural-v1.json` is absent from the repository, as the
protocol notes for the learning rate, the weight decay and the epoch count.
Batch size 64 is used, with the `FusionNetwork` defaults of 128 hidden units
and dropout 0.2, and `torch.use_deterministic_algorithms(True)` as in
`speech/scripts/train_unified.py`. The original training configuration is not
present in the repository and is on the list of questions for the author, so
these four values are this module's own and no result here should be read as
reproducing the released neural comparison.

C7. Shard division. The protocol allows splitting the work into parallel shards
if the division is recorded with the results. Rows are assigned by
`index modulo shards`, so the shards are disjoint and their union is the whole
manifest, and each shard writes its own run record under `runs/`. The division
actually used is recorded in `results/`.

C8. Note of 10 November 2026: the cross-layer tolerance, and what the ESPnet
path is not. Two points an independent review of the code raised, recorded
before any number from the test partition.

The amendment D5 says the mean half of the `all` row must agree with the mean
over layers of the per-layer pooled means "to float64 tolerance". The code uses
an absolute tolerance of 1e-3, which is the honest figure: both routes run in
the float32 the encoders produce and differ only in the order of summation, so
a float64 tolerance is not reachable in principle. Over the 20-utterance
validation sample the largest gap was 2.43e-06 for XEUS, 7.08e-06 for XLS-R
and 3.34e-06 for PhoneticXeus, three orders below the threshold. The gap of
every utterance is written to its receipt, and the largest per encoder to the
shard record, so the real distribution is in the results rather than in this
sentence. The tolerance is absolute rather than relative, which is the weaker
choice for XLS-R, whose pooled values have the largest magnitude. The observed
gaps leave enough room that tightening it was not worth invalidating a running
extraction, and the recorded gaps allow a relative bound to be checked after
the fact.

The extraction contract hashes `configs/extractor.json`, this wrapper and every
file of `speech/pronunciation/`. It does not hash the ESPnet tree that `D1`
puts on the path, so the `OOLLIM_ESPNET` variable could change the imported
ESPnet without the contract noticing. For the runs of this module the path is
`speech/resources/source/wanchichen-espnet-5b52d57`, 935 files, whose tree
digest, the sha256 of the sorted per-file sha256 listing, is
`50245e474e45f24d627d6de8cdac56ac1a684250904f09ae5b27acc5ebe89516`. Recording
it in the contract is deferred rather than refused: the change would alter the
wrapper hash and discard the extraction already in progress, so it is made
after that run finishes, and the digest above fixes what was used in the
meantime. When the contract does record it, it records as well whether
`OOLLIM_ESPNET` was set, so a run that took the default path and a run that was
pointed elsewhere are told apart rather than inferred. For the runs of this
module the variable was not set, and the default path above was used.

C9. Note of 10 November 2026: one shard of four threads rather than two of
two, and the serialization defect that preceded it. Recorded before any number
from the test partition.

The division C7 describes still holds, and the shard count actually used is one.
The first run used two shards of two threads each and began at 12.4 s per
utterance. After about 500 utterances per shard it had slowed to 21.8 s. The
cause was measured rather than guessed: the mean duration of the utterances in
that region was 3.46 s against 3.50 s at the start, so the work had not grown,
and the two processes were running at 140 and 61 per cent of one core, which is
waiting on input and output rather than computing. Two resident sets of about
3.1 GiB left roughly 6 GiB of page cache for the 5.6 GiB of weights that every
utterance reads, so the cache thrashed. One shard with four threads leaves the
cache room and runs at 8.5 s per utterance, which is faster than the two shards
were at their best.

The restart exposed a defect in this wrapper. `CROSSLAYER_INDEX` held its index
ranges as Python tuples, which `json.dumps` writes as lists, so reading
`contract.json` back and comparing it with the in-memory contract never matched
and every resumed run refused a cache this code had written. It was latent
while the first run lasted, because both shards started before the file existed.
The fix makes the ranges lists and passes the contract through the same JSON
round trip as the file before comparing, and two consecutive runs over one
output directory now both succeed where the second aborted.

The fix changes serialization and the guard that reads it, and nothing that
computes a feature: the two index values are the same integers and the slicing
they drive is identical. The identity rule of C2 runs on every utterance of the
new extraction in any case, so the features of this run are checked against the
release path row by row rather than inherited from the validation of the
previous one.

The 1,010 utterances cached by the first run were discarded. Every receipt
embeds the contract, including this wrapper's hash, so the fix invalidates them.
Rewriting those receipts would falsify provenance records, and admitting the old
hash as an accepted predecessor would weaken the integrity check permanently for
a single convenience.

C10. Note of 11 November 2026: the memory ceiling of the scope. Recorded with
the results. One shard of four threads throughout, as C9 says. The scope ran
under a 7 GiB ceiling up to row 822 of the training partition and under 11 GiB
from there on, for the rest of that partition and for the whole test partition.
Under the 7 GiB ceiling the scope sat 520 KB below its limit, so the kernel
reclaimed the page cache of the cgroup continuously, and because the extraction
call loads and deletes the three encoders per utterance, the 5.6 GiB of weights
were read from disk for every utterance. The rate was 41 to 51 s per utterance,
against 8 to 14 s after the ceiling was raised, and the input and output
pressure of the host showed half of the time with every task stalled. No feature
changes, because the identity of every utterance against the release vector is
checked per row and the ceiling is not an input of the extraction contract.
