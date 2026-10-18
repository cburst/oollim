"""Experimental blank allocation; no human boundaries or silence labels consumed."""
import numpy as np


def allocate_blanks(spans, waveform, rate=16000, cap_s=.04, energy_gate=False):
    if not np.isfinite(rate) or not np.isfinite(cap_s) or rate<=0 or cap_s<0:raise ValueError('Invalid timing configuration')
    wave=np.asarray(waveform,dtype=float)
    if wave.ndim!=1 or not len(wave) or not np.isfinite(wave).all():raise ValueError('Invalid waveform')
    duration=len(wave)/rate
    if any(not np.isfinite(a) or not np.isfinite(b) or a<0 or b<a or b>duration+1e-6 for a,b in spans):raise ValueError('Invalid span')
    if any(a[1]>b[0]+1e-6 for a,b in zip(spans,spans[1:])):raise ValueError('Overlapping spans')
    hop=max(1,round(.01*rate))
    energy=np.array([np.sqrt(np.mean(wave[i:i+hop]**2)) for i in range(0,len(wave),hop)])
    threshold=max(1e-4,float(energy.max())*10**(-35/20))
    def active(a,b):
        if b<=a:return True
        first=max(0,int(a*rate)//hop);last=min(len(energy),int(np.ceil(b*rate/hop)))
        return bool(len(energy[first:last]) and (energy[first:last]>=threshold).all())
    result=[]
    for i,(a,b) in enumerate(spans):
        left=max(0,a-cap_s, (spans[i-1][1]+a)/2 if i else 0)
        right=min(duration,b+cap_s,(b+spans[i+1][0])/2 if i+1<len(spans) else duration)
        if energy_gate:
            start,end=a,b
            while start>left+1e-9:
                candidate=max(left,start-.01)
                if not active(candidate,start):break
                start=candidate
            while end<right-1e-9:
                candidate=min(right,end+.01)
                if not active(end,candidate):break
                end=candidate
            left,right=start,end
        result.append((left,right))
    return result
