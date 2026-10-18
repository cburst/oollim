"""Offline normalized Azure label intake. Never invokes Azure or guesses export fields."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .audio import read_audio
from .encoders import sha256
from .fusion import DIMENSIONS

SCHEMA = 'icnale-sm-azure-intake-v1'


def require_text(value, name):
    if not isinstance(value,str) or not value.strip():raise ValueError(f'{name} must be nonempty text')
    return value


def validate_record(row, root):
    if row.get('schema_version')!=SCHEMA or row.get('dataset')!='icnale_sm':
        raise ValueError('Expected ICNALE SM schema; GRA is a separate dataset')
    for name in ('recording_id','speaker_id','reference_text','assessment_id'):
        require_text(row.get(name),name)
    if row.get('reference_origin') not in ('human_verbatim','reviewed_asr_verbatim'):
        raise ValueError('Require a verified verbatim transcript; no corrected prose or unreviewed ASR')
    if row.get('audio_scope')!='learner_only':raise ValueError('Learner-only audio required')
    settings=row.get('assessment',{})
    for name in ('locale','sdk_version','assessed_at','reference_text','aggregation_method'):
        require_text(settings.get(name),'assessment.'+name)
    if settings['reference_text']!=row['reference_text']:
        raise ValueError('Azure reference differs from model reference')
    if settings.get('grading_system')!='HundredMark' or settings.get('coverage')!='entire_recording':
        raise ValueError('Require documented 0–100 scores for the entire recording')
    if settings['locale']!='en-US':raise ValueError('This prepared protocol supports en-US assessments only')
    scores=row.get('scores',{})
    if set(scores)-set(DIMENSIONS):raise ValueError('Unexpected score field; do not substitute overall pronunciation for accuracy')
    values={}
    for d in DIMENSIONS:
        value=scores.get(d)
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=100):
            raise ValueError(f'Invalid {d} score')
        values[d]=None if value is None else float(value)
    if all(v is None for v in values.values()):raise ValueError('At least one dimension must be labelled')
    if values['prosody'] is not None and settings.get('prosody_enabled') is not True:
        raise ValueError('Prosody label requires enabled assessment')
    audio=(root/require_text(row.get('audio_path'),'audio_path')).resolve()
    if not audio.is_relative_to(root.resolve()):raise ValueError('Audio path escapes input root')
    digest=sha256(audio)
    if digest!=row.get('audio_sha256'):raise ValueError('Audio checksum mismatch')
    waveform=read_audio(audio)
    if len(waveform)<400:raise ValueError('Audio too short')
    pcm=hashlib.sha256(waveform.numpy().astype('<f4').tobytes()).hexdigest()
    return dict(row,scores=values,audio_path=str(audio),duration_s=len(waveform)/16000,
                decoded_sha256=pcm,source='azure',target_scale='0..100')


def import_records(source, audio_root, excluded_audio_hashes=(), excluded_decoded_hashes=()):
    records=[];ids=set();assessment_ids=set();audio_hashes=set();decoded_hashes=set()
    for line in Path(source).read_text().splitlines():
        if not line.strip():continue
        row=validate_record(json.loads(line),Path(audio_root))
        rid=row['recording_id'];aid=row['assessment_id'];digest=row['audio_sha256'];pcm=row['decoded_sha256']
        if rid in ids or aid in assessment_ids:raise ValueError('Duplicate recording or assessment ID')
        if digest in audio_hashes or pcm in decoded_hashes:raise ValueError('Duplicate audio; resolve before splitting')
        if digest in excluded_audio_hashes or pcm in excluded_decoded_hashes:raise ValueError('Audio overlaps existing development data')
        ids.add(rid);assessment_ids.add(aid);audio_hashes.add(digest);decoded_hashes.add(pcm);records.append(row)
    speakers=sorted({r['speaker_id'] for r in records},key=lambda s:hashlib.sha256(('icnale-sm-42:'+s).encode()).hexdigest())
    if len(speakers)<2:raise ValueError('Need at least two speakers for train/validation')
    nvalidation=max(1,len(speakers)//5);validation=set(speakers[:nvalidation])
    for row in records:row['split']='validation' if row['speaker_id'] in validation else 'train'
    return records


def label_matrix(records):
    return np.array([[np.nan if r['scores'][d] is None else r['scores'][d]/100 for d in DIMENSIONS] for r in records],dtype=np.float32)
