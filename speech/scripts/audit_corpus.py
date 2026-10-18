"""Audit canonical SpeechOcean files and lock speaker splits, without model metrics."""
import hashlib
import json
from pathlib import Path
import wave
import re
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'resources/canonical-metadata'
CORPUS = ROOT / 'resources/corpus/speechocean762'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    return dict(line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip())


def main():
    scores = json.loads((META / 'resource/scores.json').read_text())
    details = json.loads((META / 'resource/scores-detail.json').read_text())
    speakers = {s: table(META / s / 'utt2spk') for s in ('train', 'test')}
    train_speakers, test_speakers = [set(speakers[s].values()) for s in ('train', 'test')]
    assert not train_speakers & test_speakers
    ranked = sorted(train_speakers, key=lambda s: hashlib.sha256(('validation-v1:42:' + s).encode()).hexdigest())
    validation = set(ranked[:25])
    rows, by_hash = [], defaultdict(list)
    text_differences = []
    for split in ('train', 'test'):
        wavs = table(META / split / 'wav.scp')
        text = table(META / split / 'text')
        ages = table(META / split / 'spk2age')
        for uid, speaker in speakers[split].items():
            audio = CORPUS / wavs[uid]
            with wave.open(str(audio), 'rb') as wav:
                rate, channels, frames = wav.getframerate(), wav.getnchannels(), wav.getnframes()
                assert rate == 16000 and channels == 1 and wav.getsampwidth() == 2
                assert len(wav.readframes(frames)) == frames * channels * 2
            assert uid in scores and uid in details
            original_text = scores[uid]['text']
            if ' '.join(text[uid].split()) != ' '.join(original_text.split()):
                text_differences.append(uid)
                assert re.sub(r'[^A-Z0-9]', '', text[uid].upper()) == re.sub(r'[^A-Z0-9]', '', original_text.upper())
            digest = sha(audio)
            target_split = 'test' if split == 'test' else ('validation' if speaker in validation else 'train')
            by_hash[digest].append((uid, target_split))
            rows.append(dict(recording_id=uid, speaker_id=speaker, source_split=split,
                split=target_split, audio_path=str(audio.relative_to(ROOT)), language='en',
                reference_text=original_text, normalized_source_text=text[uid],
                age_source=ages.get(speaker), sample_rate=rate,
                duration_s=frames/rate, sha256=digest))
    assert len(rows) == len(scores) == len(details) == 5000
    duplicates = [items for items in by_hash.values() if len(items) > 1]
    cross_split = [items for items in duplicates if len({s for _, s in items}) > 1]
    if cross_split:
        raise ValueError(f'Cross-split duplicate audio: {cross_split}')
    split_data = {'seed': 42, 'policy': 'sha256-speaker-ranking-validation-v1',
                  'splits': {s: sorted(r['recording_id'] for r in rows if r['split'] == s)
                             for s in ('train', 'validation', 'test')}}
    encoded = json.dumps(split_data, sort_keys=True, indent=2) + '\n'
    lock = ROOT / 'reports/splits.json'
    if lock.exists() and lock.read_text() != encoded:
        raise ValueError('Existing split lock differs; explicit split migration required')
    lock.write_text(encoded)
    manifest = ROOT / 'resources/corpus-manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    aggregate_scales, raw_scales = {}, {}
    for key in ('accuracy', 'fluency', 'prosodic', 'completeness', 'total'):
        vals = [r[key] for r in scores.values() if key in r]
        aggregate_scales[key] = dict(min=min(vals), max=max(vals), count=len(vals))
        raw = [v for r in details.values() for v in r.get(key, [])]
        raw_scales[key] = dict(min=min(raw), max=max(raw), count=len(raw)) if raw else None
    report = dict(status='audited_not_trained', recordings=len(rows),
        source_speakers={s: len(set(v.values())) for s, v in speakers.items()},
        speaker_overlap=0, split_counts=dict(Counter(r['split'] for r in rows)),
        punctuation_or_spacing_differences=len(text_differences),
        audio_hours=sum(r['duration_s'] for r in rows)/3600,
        duration_range_s=[min(r['duration_s'] for r in rows),max(r['duration_s'] for r in rows)],
        duplicate_audio_groups=duplicates, cross_split_duplicates=cross_split,
        split_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
        manifest_sha256=sha(manifest), aggregate_observed_ranges=aggregate_scales,
        individual_rater_observed_ranges=raw_scales,
        canonical_archive_metadata_match={str(p.relative_to(META)): sha(p) == sha(CORPUS / p.relative_to(META))
            for p in META.rglob('*') if p.is_file() and not p.name.endswith('.receipt.json')
            and (CORPUS / p.relative_to(META)).is_file()})
    (ROOT / 'reports/corpus-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('aggregate_observed_ranges', 'individual_rater_observed_ranges','canonical_archive_metadata_match')}))


if __name__ == '__main__':
    main()
