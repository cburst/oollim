"""Deterministic acoustic/reference proxies, not validated diagnostic scores."""
import re
import numpy as np
from scipy.signal import correlate,find_peaks
VERSION='explicit-acoustic-reference-v1'
NAMES=('duration_s','reference_words','reference_words_per_s','active_fraction',
 'reference_words_per_active_s','internal_pause_count','internal_pause_seconds',
 'internal_pause_mean_s','internal_pause_max_s','active_run_mean_s','active_run_std_s',
 'energy_mean_dbfs','energy_std_db','energy_range_db','f0_median_hz','f0_iqr_semitones',
 'f0_std_semitones','f0_step_mean_semitones','periodic_fraction','clipping_fraction')


def runs(mask):
    changes=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)))


def extract(waveform, reference_text, rate=16000):
    x=np.asarray(waveform,dtype=float)
    if rate!=16000 or x.ndim!=1 or len(x)<640 or not np.isfinite(x).all() or not reference_text.strip():
        raise ValueError('Need finite >=40ms mono 16kHz audio and reference text')
    window,hop=640,160
    frames=np.lib.stride_tricks.sliding_window_view(x,window)[::hop]
    rms=np.sqrt(np.mean(frames**2,axis=1));db=20*np.log10(np.maximum(rms,1e-12))
    active=db>max(-45.,float(np.percentile(db,95))-30.)
    active_runs=runs(active);pauses=[]
    if active_runs:
        first,last=active_runs[0][0],active_runs[-1][1]
        pauses=[(b-a)*.01 for a,b in runs(~active[first:last]) if (b-a)*.01>=.2]
    durations=np.array([(b-a)*.01 for a,b in active_runs])
    pitches=np.full(len(frames),np.nan)
    taper=np.hanning(window)
    for i in np.flatnonzero(active):
        f=(frames[i]-frames[i].mean())*taper
        ac=correlate(f,f,mode='full',method='fft')[window-1:]
        if ac[0]<=1e-12:continue
        ac=ac/ac[0]
        # Interior local peaks avoid selecting the monotonic short-lag shoulder.
        lo,hi=rate//500,rate//60
        peaks,_=find_peaks(ac[lo-1:hi+2]);peaks=peaks+lo-1
        peaks=peaks[(peaks>=lo)&(peaks<=hi)]
        if len(peaks):
            lag=peaks[np.argmax(ac[peaks])]
            if ac[lag]>=.5:pitches[i]=rate/lag
    voiced=np.isfinite(pitches);f0=pitches[voiced]
    semi=12*np.log2(f0/100) if len(f0) else np.array([])
    adjacent=voiced[:-1]&voiced[1:]
    steps=np.abs(12*np.log2(pitches[1:][adjacent]/pitches[:-1][adjacent]))
    duration=len(x)/rate;words=len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",reference_text))
    fraction=float(active.mean());active_s=duration*fraction
    vals=[duration,words,words/duration,fraction,words/active_s if active_s else np.nan,
        len(pauses),sum(pauses),np.mean(pauses) if pauses else 0,max(pauses,default=0),
        durations.mean() if len(durations) else 0,durations.std() if len(durations) else 0,
        db.mean(),db.std(),np.percentile(db,95)-np.percentile(db,5),np.median(f0) if len(f0) else np.nan,
        np.percentile(semi,75)-np.percentile(semi,25) if len(semi) else np.nan,semi.std() if len(semi) else np.nan,
        steps.mean() if len(steps) else np.nan,voiced.mean(),np.mean(np.abs(x)>=.999)]
    return np.array(vals,dtype=np.float64)
