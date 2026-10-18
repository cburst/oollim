"""Strict importer for the existing English acoustic API CSV, not a scorer.

Requires the acoustic component's status, not just the overall API job status.
Never repair short legacy rows: omitted values shift their positional columns.
"""
import csv
import io
import math

VERSION = 'interlanguage-praat-english-summary-v2'
NAMES = (
    'number_of_syllables', 'number_of_pauses',
    'rate_of_speech (syllables/sec original duration)',
    'articulation_rate (syllables/sec speaking duration)',
    'speaking_duration (only speaking duration without pauses)',
    'original_duration (total speaking duration with pauses)',
    'balance_ratio (speaking duration)/(original duration)',
    'f0 mean', 'f0 stddev', 'f0 median', 'f0 min', 'f0 max',
    'f0quantile25', 'f0quantile75',
)
LEGACY_SCORE = 'pronunciation_posteriori_probability_score_percentage'


def parse_csv(text, *, component_status, expected_filename):
    """Return 14 ordered measurements; reject partial/invalid/ambiguous output.

    The caller must bind the CSV and component status to the same saved job,
    source audio hash and preprocessing receipt. This function cannot verify
    upstream parsing correctness or establish linguistic validity.
    """
    if component_status.get('stage') != 'complete':
        raise ValueError('Acoustic component did not complete')
    if component_status.get('metric_count') != 15:
        raise ValueError('Expected all 15 English legacy fields')
    if not component_status.get('sample_language', '').lower().startswith('en-'):
        raise ValueError('This schema is restricted to English speech')
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) != 2 or rows[0] != ['filename', *NAMES, LEGACY_SCORE]:
        raise ValueError('Unexpected CSV schema or row count')
    row = rows[1]
    if len(row) != len(rows[0]) or row[0] != expected_filename:
        raise ValueError('Incomplete row or mismatched audio identity')
    # Intentionally never parse or return the randomized legacy score.
    values = tuple(float(value) for value in row[1:-1])
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Missing, negative or nonfinite acoustic measurement')
    syllables, pauses, rate, articulation, speaking, duration, balance = values[:7]
    if syllables != int(syllables) or pauses != int(pauses):
        raise ValueError('Counts must be integers')
    if not 0 < speaking <= duration or not 0 < balance <= 1:
        raise ValueError('Invalid duration or balance')
    # All three fields are independently printed to one decimal by the pinned
    # Praat script. Compare feasible rounding intervals, not rounded quotients.
    ratio_low = max(0., speaking - .05) / (duration + .05)
    ratio_high = min(1., (speaking + .05) / max(duration - .05, 1e-12))
    if balance + .05 + 1e-12 < ratio_low or balance - .05 - 1e-12 > ratio_high:
        raise ValueError('Inconsistent speaking balance')
    mean, sd, median, low, high, q25, q75 = values[7:]
    if not (0 < low <= q25 <= median <= q75 <= high and low <= mean <= high):
        raise ValueError('Invalid pitch ordering')
    return values
