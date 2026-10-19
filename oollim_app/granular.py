"""Experimental local sentence prosody and uncalibrated word phone support."""
import gc
import json
import re
from pathlib import Path
import numpy as np

VERSION = 'sentence-three-construct-word-phone-support-v2'


def sentence_spans(text):
    # Deterministic offline Punkt rules; no inference-time resource downloads.
    from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
    params = PunktParameters()
    params.abbrev_types = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'vs', 'etc', 'e.g', 'i.e'}
    return list(PunktSentenceTokenizer(params).span_tokenize(text))


def align_ctc(logp, tokens, blank):
    """Viterbi CTC alignment; repeated phones require an intervening blank."""
    logp = np.asarray(logp, dtype=np.float64)
    tokens = np.asarray(tokens, dtype=np.int64)
    if logp.ndim != 2 or not len(tokens) or not np.isfinite(logp).all():
        raise ValueError('Invalid CTC alignment input.')
    if blank in tokens or min(tokens) < 0 or max(tokens) >= logp.shape[1]:
        raise ValueError('Reference contains an invalid phone token.')
    states = np.full(2 * len(tokens) + 1, blank, dtype=int); states[1::2] = tokens
    previous = np.full(len(states), -np.inf)
    previous[:2] = logp[0, states[:2]]
    back = np.zeros((len(logp), len(states)), dtype=np.int8)
    skip = np.zeros(len(states), dtype=bool)
    skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    for t in range(1, len(logp)):
        options = np.stack([previous, np.r_[-np.inf, previous[:-1]], np.r_[-np.inf, -np.inf, previous[:-2]]])
        options[2, ~skip] = -np.inf
        step = options.argmax(axis=0)
        previous = options[step, np.arange(len(states))] + logp[t, states]
        back[t] = step
    state = len(states)-2 + int(previous[-1] > previous[-2])
    if not np.isfinite(previous[state]):
        raise ValueError('Reference cannot be aligned to this recording; no word scores were invented.')
    path = np.empty(len(logp), dtype=int)
    for t in range(len(logp)-1, -1, -1):
        path[t] = state
        if t: state -= int(back[t,state])
    return [np.flatnonzero(path == 2*i+1) for i in range(len(tokens))]


def assess_segments(decoded, reference, bundle_path, temp, event):
    from .common import ROOT, RESOURCES, imports_path
    imports_path()
    import torch
    import soundfile as sf
    from pronunciation.audio import read_audio, batch_audio
    from pronunciation.phonetic import PhoneticXeus
    from pronunciation.text_reference import load_dictionary, text_to_phones
    from pronunciation.phone_features import reference_tokens
    from pronunciation.fusion import UnifiedScorer, DIMENSIONS
    from .engine import load_bundle
    from pronunciation.inference_memory import install
    cleanup = install()
    try:
        waveform = read_audio(decoded)
        spans = sentence_spans(reference)
        dictionary = load_dictionary(RESOURCES/'text-reference-v1/cmudict.dict')
        word_rows = []
        for sentence_index, (start,end) in enumerate(spans):
            for match in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)*|[^\W_]+", reference[start:end]):
                word_rows.append(dict(text=match.group(), char_start=start+match.start(), char_end=start+match.end(), sentence_index=sentence_index))
        event(4,'running','Aligning reference phones for word accuracy and sentence timing')
        model = PhoneticXeus(RESOURCES/'phoneticxeus').eval()
        with torch.inference_mode(): out = model(batch_audio([waveform]))[0]
        token_ids=[]; word_ranges=[]
        for word in word_rows:
            phones,status=text_to_phones(word['text'],dictionary)
            tokens,unknown=reference_tokens(phones,out.vocabulary)
            if not status['complete'] or unknown or not tokens:
                raise ValueError('Cannot align reference word: '+word['text'])
            word_ranges.append((len(token_ids),len(token_ids)+len(tokens)))
            token_ids.extend(out.vocabulary.index(t) for t in tokens)
        logp=out.logits.log_softmax(-1).cpu().numpy()
        frames=align_ctc(logp,token_ids,out.blank_id)
        # Timing is approximate: encoder stride with a half-frame endpoint.
        seconds_per_frame=out.stride_samples/16000
        duration=len(waveform)/16000
        for word,(first,last) in zip(word_rows,word_ranges):
            support=[float(np.exp(logp[frames[i],token_ids[i]]).mean()) for i in range(first,last)]
            word.update(start_seconds=max(0.,float(frames[first][0])*seconds_per_frame),
                        end_seconds=min(duration,float(frames[last-1][-1]+1)*seconds_per_frame),
                        accuracy=10*float(np.mean(support)), method='uncalibrated_aligned_phone_posterior')
        del model,out,logp;gc.collect()
        cfg=load_bundle(bundle_path)
        heads={name:UnifiedScorer(Path(bundle_path).parent/file) for name,file in cfg['heads'].items()}
        extractor=UnifiedScorer.__new__(UnifiedScorer)
        extractor.metadata=json.loads((ROOT/'configs/extractor.json').read_text())
        extractor.blocks=[name for name in ('xeus','xlsr','phoneticxeus','ipa') if any(name in head.blocks for head in heads.values())]
        extractor.widths=extractor.metadata['block_widths']
        extractor.width=sum(extractor.widths[name] for name in extractor.blocks)
        sentences=[]
        for i,(start,end) in enumerate(spans):
            indices=[j for j,w in enumerate(word_rows) if w['sentence_index']==i]
            if not indices: continue
            first,last=indices[0],indices[-1]
            a=0. if first==0 else (word_rows[first-1]['end_seconds']+word_rows[first]['start_seconds'])/2
            b=duration if last==len(word_rows)-1 else (word_rows[last]['end_seconds']+word_rows[last+1]['start_seconds'])/2
            sentence=dict(text=reference[start:end],char_start=start,char_end=end,start_seconds=a,end_seconds=b)
            clip=Path(temp)/f'sentence-{i}.wav'
            try:
                if b-a<.25: raise ValueError('Aligned sentence is shorter than 0.25 seconds.')
                sf.write(clip,waveform[int(a*16000):int(b*16000)].numpy(),16000,subtype='FLOAT')
                event(5,'running',f'Scoring accuracy, fluency and prosody for sentence {i+1}/{len(spans)}')
                features,_=extractor.extract(clip,sentence['text'],RESOURCES)
                blocks={};offset=0
                for name in extractor.blocks:
                    blocks[name]=features[:,offset:offset+extractor.widths[name]];offset+=extractor.widths[name]
                for name,head in heads.items():
                    values=np.concatenate([blocks[block] for block in head.blocks],axis=1)
                    sentence[name]=float(head.predict_features(values)[0,DIMENSIONS.index(name)])
                sentence['status']='completed'
            except Exception as exc:
                sentence.update(accuracy=None,fluency=None,prosody=None,status='failed',error=str(exc))
            finally: clip.unlink(missing_ok=True)
            sentences.append(sentence)
        return dict(version=VERSION,words=word_rows,sentences=sentences,
                    word_score_status='Experimental phone-support scores, not calibrated human word-accuracy predictions.',
                    sentence_score_status='Utterance accuracy, fluency and prosody ridges applied to automatically aligned sentence audio; segment validation pending.')
    finally: cleanup.remove()
