"""Fixed, uncalibrated phone evidence for controlled read-aloud experiments.

Canonical dictionary phones are inputs, never human perceived-phone or score labels.
IPA matches are approximate diagnostics, not automatically pronunciation errors.
"""
from collections import defaultdict
import re
import numpy as np
import torch
from torch.nn import functional as F

MAPPING_VERSION = 'speechocean-arpabet-approx-ipa-v1'
# Explicitly versioned independently of the L2-ARCTIC annotation parser.
IPA = dict(AA=['ɑ'], AE=['æ'], AH=['ʌ'], AO=['ɔ'], AW=['a','ʊ'], AY=['a','ɪ'],
 B=['b'], CH=['t͡ʃ'], D=['d'], DH=['ð'], EH=['ɛ'], ER=['ɜ˞'], EY=['e','ɪ'],
 F=['f'], G=['ɡ'], HH=['h'], IH=['ɪ'], IY=['i'], JH=['d͡ʒ'], K=['k'], L=['l'],
 M=['m'], N=['n'], NG=['ŋ'], OW=['o','ʊ'], OY=['ɔ','ɪ'], P=['p'], R=['ɹ'],
 S=['s'], SH=['ʃ'], T=['t'], TH=['θ'], UH=['ʊ'], UW=['u'], V=['v'], W=['w'],
 Y=['j'], Z=['z'], ZH=['ʒ'])
FEATURE_NAMES = ('entropy_mean','entropy_std','max_probability_mean',
 'blank_probability_mean','greedy_blank_fraction','greedy_tokens_per_second',
 'reference_tokens_per_second','reference_coverage','reference_complete',
 'greedy_reference_length_ratio','edit_rate','substitution_rate','deletion_rate',
 'insertion_rate','ctc_alignable','ctc_nll_per_frame','ctc_nll_per_reference_token')


def canonical_references(path, allowed_ids):
    words = defaultdict(dict)
    for line in path.read_text().splitlines():
        key, phones = line.replace('\\t','\t').split('\t')
        utterance, word = key.rsplit('.',1)
        if utterance not in allowed_ids:
            continue
        index = int(word)
        if index in words[utterance]:
            raise ValueError('Duplicate canonical word')
        words[utterance][index] = [re.sub(r'_[BIES]$', '', p) for p in phones.split()]
    result = {}
    for utterance in allowed_ids:
        entries = words[utterance]
        if not entries or sorted(entries)!=list(range(len(entries))):
            raise ValueError('Missing/discontinuous canonical reference')
        result[utterance] = [p for i in sorted(entries) for p in entries[i]]
    return result


def reference_tokens(phones, vocabulary):
    mapped, unknown = [], []
    for phone in phones:
        base = re.sub('[012]$', '', phone)
        tokens = ['ə'] if phone=='AH0' else IPA.get(base)
        if tokens is None or any(t not in vocabulary for t in tokens):
            unknown.append(phone)
        else:
            mapped.extend(tokens)
    return mapped, unknown


def edit_counts(reference, observed):
    """Levenshtein counts, fixed diagonal/deletion/insertion tie order."""
    previous = [(j,0,0,j) for j in range(len(observed)+1)]
    for i, ref in enumerate(reference,1):
        current = [(i,0,i,0)]
        for j, obs in enumerate(observed,1):
            distance,s,d,ins = previous[j-1]
            mismatch = int(ref!=obs)
            diagonal = (distance+mismatch,s+mismatch,d,ins)
            distance,s,d,ins = previous[j]
            deletion = (distance+1,s,d+1,ins)
            distance,s,d,ins = current[-1]
            insertion = (distance+1,s,d,ins+1)
            current.append(min((diagonal,deletion,insertion),key=lambda x:x[0]))
        previous = current
    return previous[-1]


def phone_evidence(logits, vocabulary, blank_id, canonical_phones, duration_s):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    if logits.ndim!=2 or not len(logits) or logits.shape[1]!=len(vocabulary) or not torch.isfinite(logits).all():
        raise ValueError('Invalid phone logits')
    if not 0<=blank_id<len(vocabulary) or not np.isfinite(duration_s) or duration_s<=0:
        raise ValueError('Invalid blank ID or duration')
    tokens, unknown = reference_tokens(canonical_phones, vocabulary)
    complete = bool(tokens) and not unknown
    logp = logits.log_softmax(-1); prob = logp.exp()
    entropy = -(prob*logp).sum(-1)
    path = logits.argmax(-1)
    greedy = [vocabulary[i] for i in path.unique_consecutive().tolist() if i!=blank_id]
    denominator = max(len(tokens),1)
    edits = edit_counts(tokens,greedy) if complete else (0,0,0,0)
    nll, alignable = 0., False
    if complete:
        ids = torch.tensor([vocabulary.index(t) for t in tokens],dtype=torch.long)
        value = F.ctc_loss(logp[:,None,:],ids,torch.tensor([len(logits)]),torch.tensor([len(ids)]),
            blank=blank_id,reduction='sum',zero_infinity=False)
        alignable = bool(torch.isfinite(value))
        if alignable:
            nll = float(value)
    values = [float(entropy.mean()),float(entropy.std(unbiased=False)),
        float(prob.max(-1).values.mean()),float(prob[:,blank_id].mean()),
        float((path==blank_id).float().mean()),len(greedy)/duration_s,
        len(tokens)/duration_s,1-len(unknown)/max(len(canonical_phones),1),float(complete),
        len(greedy)/denominator if complete else 0.,*[e/denominator for e in edits],
        float(alignable),nll/len(logits),nll/denominator]
    vector = np.array(values,dtype=np.float64)
    if vector.shape!=(len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError('Invalid diagnostic vector')
    return vector, dict(reference_complete=complete,ctc_alignable=alignable,
        unknown_phones=unknown,reference_tokens=len(tokens),greedy_tokens=len(greedy))
