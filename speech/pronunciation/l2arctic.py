"""L2-ARCTIC-specific annotation parsing and explicit approximate IPA mapping."""
import re

# Deliberately corpus-local. Diphthongs/affricates may occupy multiple model tokens.
IPA = dict(AA=['ɑ'], AE=['æ'], AH=['ʌ'], AO=['ɔ'], AW=['a','ʊ'], AX=['ə'],
 AY=['a','ɪ'], B=['b'], CH=['t͡ʃ'], D=['d'], DH=['ð'], EH=['ɛ'], ER=['ɜ˞'],
 EY=['e','ɪ'], F=['f'], G=['ɡ'], HH=['h'], IH=['ɪ'], IY=['i'], JH=['d͡ʒ'],
 K=['k'], L=['l'], M=['m'], N=['n'], NG=['ŋ'], OW=['o','ʊ'], OY=['ɔ','ɪ'],
 P=['p'], R=['ɹ'], S=['s'], SH=['ʃ'], T=['t'], TH=['θ'], UH=['ʊ'], UW=['u'],
 V=['v'], W=['w'], Y=['j'], Z=['z'], ZH=['ʒ'])
SILENCE={'','sil','sp','spn'}
MAPPING_VERSION='l2-arctic-arpabet-to-ipa-approx-v1'


def phone_tokens(label, vocabulary):
    if label.lower() in SILENCE:return [],None
    if '*' in label or label.lower()=='err':return None,'deviation_or_unknown'
    base=re.sub('[012]$','',label.upper())
    tokens=IPA.get(base)
    # Preserve lexical stress in raw labels; recognize reduced unstressed AH.
    if label.upper()=='AH0':tokens=['ə']
    if tokens is None:return None,'unmapped_phone'
    if any(t not in vocabulary for t in tokens):return None,'absent_model_token'
    return tokens,None


def parse_phone_label(raw):
    pieces=[x.strip() for x in raw.split(',')]
    if len(pieces)==3:pieces[2]=pieces[2].lower()
    if len(pieces)==1:return dict(canonical=pieces[0],perceived=pieces[0],operation='correct',raw=raw)
    if len(pieces)!=3 or pieces[2] not in ('s','a','d'):
        raise ValueError(f'Unknown annotation grammar: {raw!r}')
    return dict(canonical=pieces[0],perceived=pieces[1],operation=pieces[2],raw=raw)


def read_phone_tier(text):
    if 'Object class = "TextGrid"' not in text:raise ValueError('Expected long TextGrid')
    tiers=re.split(r'\bitem\s*\[\d+\]\s*:',text)[1:]
    found=[]
    for tier in tiers:
        name=re.search(r'\bname\s*=\s*"((?:[^"]|"")*)"',tier)
        if not name or name.group(1)!='phones':continue
        expected=int(re.search(r'intervals:\s*size\s*=\s*(\d+)',tier).group(1))
        pattern=r'intervals\s*\[\d+\]\s*:\s*xmin\s*=\s*([\d.eE+\-]+)\s*xmax\s*=\s*([\d.eE+\-]+)\s*text\s*=\s*"((?:[^"]|"")*)"'
        rows=[dict(start=float(a),end=float(b),**parse_phone_label(label.replace('""','"'))) for a,b,label in re.findall(pattern,tier)]
        if len(rows)!=expected:raise ValueError('Incomplete phone tier parse')
        if any(r['start']<0 or r['end']<r['start'] for r in rows):raise ValueError('Invalid interval')
        if any(a['end']>b['start']+1e-6 for a,b in zip(rows,rows[1:])):raise ValueError('Overlapping intervals')
        found.append(rows)
    if len(found)!=1:raise ValueError('Expected one phones tier')
    return found[0]


def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):row.append(min(row[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=row
    return prev[-1]


def phone_tokens_with_deviation(label, vocabulary):
    """Retain deviation flag; base-phone conditioning is approximate, not recognition."""
    if '*' not in label:
        tokens,reason=phone_tokens(label,vocabulary)
        return tokens,reason,False
    # Only the documented suffix marker is supported, never arbitrary stripping.
    if not re.fullmatch(r'[A-Za-z]+[012]?\*',label):
        return None,'unsupported_deviation_syntax',True
    tokens,reason=phone_tokens(label[:-1],vocabulary)
    return tokens,reason,True
