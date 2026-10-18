"""Frozen whole-utterance Stella inference; no fitting or sentence splitting."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
from .encoders import ROOT, sha256
from .fusion import UnifiedScorer, DIMENSIONS

CONFIG = ROOT/'experiments/stella-api-v2.json'
RIDGE_CONFIGS = {'speechocean': CONFIG, **{name: ROOT/'experiments'/('stella-api-'+name+'-seven-v1.json') for name in ['korean', 'aesop', 'koreanocean', 'aesopocean', 'koreanaesop', 'aesopkoreanocean']}}


def configuration(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    if hashlib.sha256(cfg['reference_text'].encode()).hexdigest() != cfg['reference_sha256']:
        raise ValueError('Reference checksum mismatch')
    return cfg


def checked_scorer(asset):
    path = ROOT/asset['path']
    if sha256(path) != asset['sha256'] or sha256(path.with_suffix('.json')) != asset['metadata_sha256']:
        raise ValueError('Pinned scoring asset changed')
    return UnifiedScorer(path)


def decode_audio(source, destination, max_seconds=180):
    # Decode only: preserve native channels/rate for the training pipeline's
    # arithmetic mono average and scipy polyphase resampling. No silence trimming.
    subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error',
        '-protocol_whitelist','file,pipe',
        '-format_whitelist','wav,mp3,flac,ogg,mov,matroska,webm,aac,aiff,asf,amr',
        '-i',str(source),'-map','0:a:0','-vn','-t',str(max_seconds+1),
        '-c:a','pcm_f32le','-fs','134217728','-f','wav','-y',str(destination)],
        stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=90,check=True)
    if Path(destination).stat().st_size >= 134152192:
        raise ValueError('Decoded audio exceeds safe size limit')
    info = sf.info(destination)
    duration = info.frames/info.samplerate
    if not .25 <= duration <= max_seconds or not 1 <= info.channels <= 8:
        raise ValueError(f'Audio must be 0.25–{max_seconds} seconds and have at most 8 channels')
    return duration


def score_whole(source, workdir, ridge='speechocean'):
    config_path = RIDGE_CONFIGS[ridge]
    cfg = configuration(config_path)
    decoded = Path(workdir)/'decoded.wav'
    limits=json.loads((ROOT/'experiments/stella-api-limits-v1.json').read_text())
    duration = decode_audio(source, decoded, limits['max_duration_seconds'])
    extractor = checked_scorer(cfg['extractor'])
    features, status = extractor.extract(decoded, cfg['reference_text'])
    if not status['text']['complete'] or not status['phone']['reference_complete']:
        raise ValueError('Incomplete reference representation')
    blocks = {}; offset = 0
    for block in extractor.blocks:
        width = extractor.widths[block]
        blocks[block] = features[:, offset:offset+width]; offset += width
    scores = {}; models = {}
    for index, construct in enumerate(DIMENSIONS):
        scorer = checked_scorer(cfg['heads'][construct])
        joined = np.concatenate([blocks[name] for name in scorer.blocks],axis=1)
        scores[construct] = float(scorer.predict_features(joined)[0,index])
        models[construct] = dict(blocks=scorer.blocks, checkpoint_sha256=cfg['heads'][construct]['sha256'])
    return dict(schema_version=cfg['schema_version'], model_version=cfg['model_version'],
        configuration_sha256=sha256(config_path), ridge=cfg['ridge'],reference_id=cfg['reference_id'],
        reference_sha256=cfg['reference_sha256'],duration_seconds=duration,input_limits_version=limits['limits_version'],
        scores=scores,construct_models=models,reference_status=status,
        score_scale='0–10; raw regression outputs are not clipped',
        processing='Whole recording; mono average and 16kHz polyphase resampling; no splitting, trimming or chunking',
        limitations=['passage_length_generalization_unvalidated','exploratory_development_selection',
                    'phoneticxeus_pretraining_exposure_unresolved'] + (['korean_endpoint_mapping_rubric_equivalence_unvalidated','korean_fluency_prosody_share_label'] if ridge in ('korean','koreanocean','koreanaesop','aesopkoreanocean') else []))
