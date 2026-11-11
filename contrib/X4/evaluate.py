"""Evaluation of the released SpeechOcean762 heads on the official test partition.

Runs S1 to S4 of contrib/X4/PROTOCOL.md, computes the metrics with speaker
bootstrap intervals, and writes the tables and the predictions to
contrib/X4/results/ so the tables can be rebuilt without extracting again.

The test partition informs no choice. Every penalty, block set, seed and epoch
rule is either read from a released checkpoint or chosen on the 500 development
utterances of the existing project split.
"""
import argparse,itertools,json,sys,time
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,rankdata,spearmanr

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'speech')]
from reproduce.train import (save,digest,BLOCKS,CONSTRUCTS,ALPHAS,weighted_fit,
                             select_candidate,load_corpus,metrics as project_metrics)

RULES=(('selection-comparison-v1','spearman'),('selection-comparison-v1','damae'),
       ('selection-comparison-v1','historical_grid_damae'),
       ('banded-3ratio-v1','banded_3ratio_spearman'),
       ('banded-3ratio-v1','banded_3ratio_accuracy_mae'))
# phone_evidence reads the PhoneticXeus logits, so the ipa block carries the same
# upstream exposure as the phoneticxeus block even though the protocol names only
# the latter. Recorded as a posterior refinement in PROTOCOL.md.
EXPOSED_BLOCKS=('phoneticxeus','ipa')
METRIC_NAMES=('pearson','spearman','mae','pearson_clipped')
# The released application default. The exposed side of the exposure pairing is
# this rule, fixed in advance, rather than whichever rule the test partition
# happens to favour.
PRESPECIFIED_RULE='spearman'
S4_DEFAULTS={'learning_rate':1e-3,'weight_decay':1e-2,'epochs':300,'batch_size':64,'seeds':(0,1,2,3,4)}

def load_test(path):
 """load_corpus refuses the test partition by design; the same checks, no split."""
 path=Path(path);manifest=json.loads(path.read_text());rows=manifest['rows']
 if manifest['schema_version']!='speech-feature-cache-v1' or manifest['target_order']!=list(CONSTRUCTS):raise ValueError('Unsupported manifest')
 if manifest['target_scale']!='0..10':raise ValueError('Map labels before evaluation')
 if not rows or len({r['recording_id'] for r in rows})!=len(rows):raise ValueError('Duplicate/empty recording IDs')
 if any(r['split']!='test' for r in rows):raise ValueError('Expected the official test partition only')
 if any(not r['speaker_id'] for r in rows):raise ValueError('Missing speaker')
 y=np.asarray([r['targets'] for r in rows],float)
 if y.shape!=(len(rows),3) or not np.isfinite(y).all() or np.any((y<0)|(y>10)):raise ValueError('Invalid targets')
 fp=path.parent/manifest['features']
 if digest(fp)!=manifest['features_sha256']:raise ValueError('Feature checksum mismatch')
 with np.load(fp,allow_pickle=False) as f:
  if list(f['recording_ids'].astype(str))!=[r['recording_id'] for r in rows]:raise ValueError('Feature row order mismatch')
  features={k:f[k].astype(float) for k in BLOCKS}
 if any(v.ndim!=2 or len(v)!=len(rows) or not np.isfinite(v).all() for v in features.values()):raise ValueError('Invalid features')
 return {'rows':rows,'features':features,'targets':y,'manifest_sha256':digest(path)}

def design(features,blocks):return np.concatenate([features[k] for k in blocks],1)
def gains(features,blocks):return np.concatenate([np.full(features[k].shape[1],np.sqrt(2048/features[k].shape[1])) for k in blocks])
def exposed(blocks):return any(b in EXPOSED_BLOCKS for b in blocks)

def fast_pearson(a,b):
 # float64 throughout: the S4 network returns float32, and accumulating in it
 # loses enough precision to disagree with scipy.
 a=np.asarray(a,float);b=np.asarray(b,float)
 a=a-a.mean();b=b-b.mean();d=np.sqrt(float((a*a).sum())*float((b*b).sum()))
 return float((a*b).sum()/d) if d>0 else float('nan')

def point_metrics(y,p,yrank=None):
 y=np.asarray(y,float);p=np.asarray(p,float);clipped=np.clip(p,0,10)
 yr=rankdata(y) if yrank is None else yrank
 return {'pearson':fast_pearson(y,p),'spearman':fast_pearson(yr,rankdata(p)),
         'mae':float(np.abs(y-p).mean()),'pearson_clipped':fast_pearson(y,clipped),'n':int(len(y))}

def speaker_resamples(speakers,reps,seed):
 """Cluster bootstrap over speakers. Resample sizes vary when a coverage rule
 drops utterances, so the index arrays are kept as a list rather than a matrix."""
 speakers=np.asarray(speakers);uniq=sorted(set(speakers.tolist()))
 index=[np.flatnonzero(speakers==s) for s in uniq]
 rng=np.random.default_rng(seed)
 draws=rng.integers(0,len(uniq),(reps,len(uniq)))
 return [np.concatenate([index[k] for k in row]) for row in draws]

def intervals(y,p,samples,yranks):
 collected={k:np.empty(len(samples)) for k in METRIC_NAMES}
 for i,idx in enumerate(samples):
  m=point_metrics(y[idx],p[idx],yranks[i])
  for k in METRIC_NAMES:collected[k][i]=m[k]
 return {k:{'low':float(np.nanquantile(v,.025)),'high':float(np.nanquantile(v,.975)),
            'nan_resamples':int(np.isnan(v).sum())} for k,v in collected.items()}

def s1(test):
 """The released heads, exactly as distributed, through the release scorer."""
 from pronunciation.fusion import UnifiedScorer
 for family,rule in RULES:
  for j,construct in enumerate(CONSTRUCTS):
   path=ROOT/'models'/family/rule/'speechocean'/f'{construct}.npz'
   scorer=UnifiedScorer(path)
   pred=scorer.predict_features(design(test['features'],scorer.blocks))[:,j]
   yield dict(system='S1',rule=rule,construct=construct,blocks=list(scorer.blocks),
              alpha=scorer.metadata['training']['alpha'],source=str(path.relative_to(ROOT)),
              checkpoint_sha256=scorer.metadata['checkpoint_sha256']),pred

def s2(train,test):
 """The same blocks and the same penalty, refitted on all 2,500 train rows."""
 for family,rule in RULES:
  for j,construct in enumerate(CONSTRUCTS):
   meta=json.loads((ROOT/'models'/family/rule/'speechocean'/f'{construct}.json').read_text())
   blocks=list(meta['blocks']);alpha=meta['training']['alpha']
   # weighted_fit takes one scalar penalty. A per-block ratio in a future
   # release would make this refit silently wrong, so it is refused here.
   ratio=meta['training'].get('penalty_ratio_xlsr_over_xeus',meta.get('penalty_ratio_xlsr_over_xeus'))
   if ratio not in (None,1,1.0):
    raise ValueError(f'{rule}/{construct} carries penalty ratio {ratio}; the S2 refit uses one scalar penalty')
   xt=design(train['features'],blocks);gain=gains(train['features'],blocks)
   w=np.full(len(xt),1/len(xt))
   mean,scale,intercept,coefs=weighted_fit(xt,train['targets'],w,[alpha],gain)
   pred=(((design(test['features'],blocks)-mean)/scale*gain)@coefs[alpha]+intercept)[:,j]
   yield dict(system='S2',rule=rule,construct=construct,blocks=blocks,alpha=alpha,
              fit_rows=int(len(xt)),source=str((ROOT/'models'/family/rule/'speechocean'/f'{construct}.json').relative_to(ROOT))),pred

def s3(train,test):
 """One ridge per block combination, penalty chosen on the 500 development rows."""
 mask=train['train'];y=train['targets']
 for size in range(1,len(BLOCKS)+1):
  for combination in itertools.combinations(BLOCKS,size):
   blocks=list(combination)
   xt=design(train['features'],blocks);gain=gains(train['features'],blocks)
   w=np.full(int(mask.sum()),1/int(mask.sum()))
   mean,scale,intercept,coefs=weighted_fit(xt[mask],y[mask],w,ALPHAS,gain)
   zdev=(xt[~mask]-mean)/scale*gain
   ztest=(design(test['features'],blocks)-mean)/scale*gain
   for j,construct in enumerate(CONSTRUCTS):
    candidates=[]
    for alpha,coef in coefs.items():
     scores=project_metrics(y[~mask,j],(zdev@coef+intercept)[:,j])
     if scores['spearman'] is None:raise ValueError('Undefined selection Spearman')
     candidates.append({'alpha':alpha,'spearman':float(scores['spearman']),'mae':float(scores['mae'])})
    best=select_candidate(candidates,'spearman')
    pred=(ztest@coefs[best['alpha']]+intercept)[:,j]
    yield dict(system='S3',rule='development-selected',construct=construct,blocks=blocks,
               alpha=best['alpha'],development_spearman=best['spearman'],
               development_mae=best['mae'],fit_rows=int(mask.sum()),
               development_rows=int((~mask).sum())),pred

def s4(train,test,config,sets_for_construct):
 """The repository fusion network on the S1 block sets, five seeds.

 The released heads use two block sets, and their scaler_mean, scaler_scale and
 block_gain are bitwise identical across the five selection rules for a given
 set, so the standardized design matrix the network reads depends on the block
 set alone. Two networks therefore cover all five rules.
 """
 import copy,torch
 from pronunciation.fusion import FusionNetwork,UnifiedScorer,DIMENSIONS
 from pronunciation.scoring import masked_loss
 torch.use_deterministic_algorithms(True);torch.set_num_threads(config['threads'])
 sets={}
 for family,rule in RULES:
  for construct in CONSTRUCTS:
   path=ROOT/'models'/family/rule/'speechocean'/f'{construct}.npz'
   scorer=UnifiedScorer(path);key=tuple(scorer.blocks)
   if key not in sets:sets[key]=(scorer,str(path.relative_to(ROOT)))
   else:
    reference=sets[key][0]
    for k in ('scaler_mean','scaler_scale','block_gain'):
     if not np.array_equal(scorer.arrays[k],reference.arrays[k]):
      raise ValueError(f'Standardization differs across rules for {key}; S4 needs one network per rule')
 mask=train['train'];y=train['targets']
 s1_sets={c:tuple(sets_for_construct[c]) for c in sets_for_construct}
 for blocks,(scorer,source) in sets.items():
  blocks=list(blocks)
  xt=design(train['features'],blocks)
  z=(xt-scorer.arrays['scaler_mean'])/scorer.arrays['scaler_scale']*scorer.arrays['block_gain']
  ztest=(design(test['features'],blocks)-scorer.arrays['scaler_mean'])/scorer.arrays['scaler_scale']*scorer.arrays['block_gain']
  xtrain=torch.from_numpy(z[mask].astype(np.float32));xdev=torch.from_numpy(z[~mask].astype(np.float32))
  xev=torch.from_numpy(ztest.astype(np.float32))
  ytrain=torch.from_numpy((y[mask]/10).astype(np.float32))
  for seed in config['seeds']:
   torch.manual_seed(seed);model=FusionNetwork(z.shape[1])
   optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
   best=float('inf');beststate=None;bestepoch=None
   for epoch in range(1,config['epochs']+1):
    model.train()
    for ids in torch.randperm(len(xtrain)).split(config['batch_size']):
     optimizer.zero_grad(set_to_none=True)
     loss=masked_loss(model(xtrain[ids]),{'speechocean':{d:ytrain[ids,k] for k,d in enumerate(DIMENSIONS)}})
     if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
     loss.backward()
     # Parity with speech/scripts/train_unified.py, which refuses a missing or
     # nonfinite gradient rather than stepping on it.
     if not all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters()):
      raise ValueError('Missing or nonfinite gradient')
     optimizer.step()
    model.eval()
    with torch.no_grad():dev=model.array(xdev).numpy()
    # The protocol fixes the epoch rule: accuracy mean absolute error on dev.
    objective=float(np.abs(dev[:,0]-y[~mask,0]).mean())
    if objective<best:best=objective;bestepoch=epoch;beststate=copy.deepcopy(model.state_dict())
   model.load_state_dict(beststate);model.eval()
   with torch.no_grad():pred=model.array(xev).numpy()
   for j,construct in enumerate(CONSTRUCTS):
    # One network predicts the three constructs, so each block set yields a row
    # for all three. Only the construct whose released head uses this block set
    # belongs to the S1 pairing; the other two are reported and marked.
    yield (dict(system='S4',rule=f'seed-{seed}',construct=construct,blocks=blocks,alpha=None,
               in_s1_pairing=(tuple(blocks)==s1_sets.get(construct)),
               seed=int(seed),best_epoch=int(bestepoch),development_accuracy_mae=best,
               scaler_source=source,architecture={'hidden':128,'dropout':.2},
               hyperparameters={k:config[k] for k in ('learning_rate','weight_decay','epochs','batch_size')}),pred[:,j])

LITERATURE=(('GOPT','fluency 0.753, prosodic 0.760'),('HMamba','fluency 0.848, prosodic 0.843'))

def cell(coverage,metric,places=4):
 value=coverage[metric];bounds=coverage['interval'][metric]
 return f"{value:.{places}f} [{bounds['low']:.{places}f}, {bounds['high']:.{places}f}]"

def table(lines,rows,columns,header):
 lines.append('| '+' | '.join(header)+' |')
 lines.append('|'+'|'.join('---' for _ in header)+'|')
 for r in rows:lines.append('| '+' | '.join(columns(r))+' |')
 lines.append('')

def write_tables(records,run,path):
 """Human-readable tables. metrics.json keeps every figure and every interval."""
 by=lambda system:[r for r in records if r['system']==system]
 flag=lambda r:'yes' if r['potentially_optimistic'] else 'no'
 L=['# X4 results: the official SpeechOcean762 test partition','',
  'Written by `contrib/X4/evaluate.py`. Every figure and every interval is in',
  '`metrics.json`. The predictions are in `predictions.npz`, so these tables can',
  'be rebuilt without extracting again.','',
  f"Resamples {run['resamples']}, seed {run['seed']}. Coverage: "
  +', '.join(f'{k} {v}' for k,v in run['coverage'].items())+'.','',
  'Intervals are 95 per cent speaker cluster bootstrap intervals. The column',
  '"exposed" marks a system whose blocks include `phoneticxeus` or `ipa`. The',
  'released PhoneticXeus configuration lists a development set whose transcripts',
  'come from a split named after this test partition, and `phone_evidence`',
  'derives the `ipa` block from the same checkpoint. Those rows are potentially',
  'optimistic.','']
 for system,title in (('S1','S1. The released heads, exactly as distributed'),
                      ('S2','S2. The same blocks and penalties refitted on all 2,500 train rows')):
  rows=by(system)
  if not rows:continue
  L.append(f'## {title}');L.append('')
  for name in run['coverage']:
   L.append(f'### {name}');L.append('')
   table(L,rows,lambda r,n=name:[r['rule'],r['construct'],'+'.join(r['blocks']),str(r['alpha']),
     cell(r['coverage'][n],'pearson'),cell(r['coverage'][n],'spearman'),
     cell(r['coverage'][n],'mae',3),f"{r['coverage'][n]['pearson_clipped']:.4f}",flag(r)],
     ['rule','construct','blocks','penalty','Pearson [95% CI]','Spearman [95% CI]','MAE [95% CI]','Pearson clipped','exposed'])
 rows=by('S3')
 if rows:
  L.append('## S3. One ridge per block combination, penalty chosen on the 500 development rows');L.append('')
  L.append('Descriptive by construction. The penalty is the only quantity selected,')
  L.append('and it is selected on the development split, never on the test partition.');L.append('')
  for name in run['coverage']:
   L.append(f'### {name}');L.append('')
   for construct in CONSTRUCTS:
    # Ordered by the development criterion, not by the test partition, so the
    # table does not read as a ranking produced by the test.
    sub=sorted((r for r in rows if r['construct']==construct),
               key=lambda r:(-r['development_spearman'],r['development_mae']))
    L.append(f'**{construct}**');L.append('')
    table(L,sub,lambda r,n=name:['+'.join(r['blocks']),str(r['alpha']),
      f"{r['development_spearman']:.4f}",cell(r['coverage'][n],'pearson'),
      cell(r['coverage'][n],'spearman'),cell(r['coverage'][n],'mae',3),flag(r)],
      ['blocks','penalty','development Spearman','Pearson [95% CI]','Spearman [95% CI]','MAE [95% CI]','exposed'])
 rows=by('S4')
 if rows:
  L.append('## S4. The repository fusion network on the S1 block sets, five seeds');L.append('')
  L.append('Two networks cover the five selection rules. The `scaler_mean`,')
  L.append('`scaler_scale` and `block_gain` of the released heads are bitwise')
  L.append('identical across the five rules for a given block set, so the')
  L.append('standardized design matrix the network reads depends on the block set')
  L.append('alone, and the five rules differ only in the ridge penalty, which a')
  L.append('network never uses. **The five rule rows of one block set therefore share')
  L.append('a single network and are not five independent results.** Each row below is')
  L.append('one block set, not one rule. `evaluate.py` asserts the identity and')
  L.append('raises if a future release breaks it.');L.append('')
  L.append('Mean and standard deviation across the five seeds, as the protocol')
  L.append('fixes. The epoch of every network is chosen by the accuracy mean')
  L.append('absolute error on the development split, which is what the protocol and')
  L.append('speech/scripts/train_unified.py fix, including for the network whose')
  L.append('released constructs are fluency and prosody. The column "in S1')
  L.append('pairing" marks the one construct per block set whose released head')
  L.append('uses that block set. The other two rows of a network are reported but')
  L.append('are not part of the exposure pairing.');L.append('')
  for name in run['coverage']:
   L.append(f'### {name}');L.append('')
   groups={}
   for r in rows:groups.setdefault(('+'.join(r['blocks']),r['construct']),[]).append(r)
   summary=[]
   for (blocks,construct),group in sorted(groups.items()):
    stats={}
    for metric in ('pearson','spearman','mae'):
     v=np.array([g['coverage'][name][metric] for g in group],float)
     stats[metric]=(float(v.mean()),float(v.std(ddof=1)))
    summary.append({'blocks':blocks,'construct':construct,'stats':stats,
      'epochs':[g['best_epoch'] for g in group],'exposed':group[0]['potentially_optimistic'],
      'in_pairing':bool(group[0].get('in_s1_pairing'))})
   table(L,summary,lambda r:[r['blocks'],r['construct'],
     f"{r['stats']['pearson'][0]:.4f} ± {r['stats']['pearson'][1]:.4f}",
     f"{r['stats']['spearman'][0]:.4f} ± {r['stats']['spearman'][1]:.4f}",
     f"{r['stats']['mae'][0]:.3f} ± {r['stats']['mae'][1]:.3f}",
     ', '.join(str(e) for e in r['epochs']),'yes' if r['exposed'] else 'no',
     'yes' if r['in_pairing'] else 'no'],
     ['blocks','construct','Pearson mean ± sd','Spearman mean ± sd','MAE mean ± sd','best epochs','exposed','in S1 pairing'])
 L.append('## Upstream exposure, reported side by side');L.append('')
 L.append('Both sides of the pairing are fixed in advance, so that neither side')
 L.append('gains the advantage of having been picked by the test partition.')
 L.append('')
 L.append('The exposed side is the released application default, the `spearman`')
 L.append('rule, for S1 and S2. For S3 it is the exposed block combination chosen')
 L.append('by the same rule the clean side uses, the highest development Spearman')
 L.append('correlation with ties broken by the lower development mean absolute')
 L.append('error. For S4 it is the network whose block set is the one the released')
 L.append('head uses for that construct, reported as mean and standard deviation')
 L.append('over the five seeds.')
 L.append('')
 L.append('The clean side is the S3 system whose blocks contain neither')
 L.append('`phoneticxeus` nor `ipa`, chosen by the rule module X6 uses: the highest')
 L.append('development Spearman correlation, ties broken by the lower development')
 L.append('mean absolute error. X6 averages that Spearman over three corpora with')
 L.append('equal weights and reports a development-size-weighted variant beside')
 L.append('it. S3 fits on one corpus and selects on one development set of 500')
 L.append('utterances, so the mean has a single term and the two weightings select')
 L.append('the same system. There is no second row to report.')
 L.append('')
 L.append('For fluency and prosody the released heads, under the `spearman` rule,')
 L.append('use XEUS and XLS-R, which carry no exposure. Those two constructs')
 L.append('therefore have no exposed S1 or S2 row, and their S4 network of the S1')
 L.append('block set is not exposed either. Their pairing shows the clean')
 L.append('reference, the exposed S3 system chosen on development data, and the')
 L.append('descriptive rows.')
 L.append('')
 L.append('Below the pre-specified pair, the system that is best by test Pearson')
 L.append('correlation is given for each side, marked descriptive, because that')
 L.append('figure does involve the test partition and carries the advantage of')
 L.append('having been chosen on it.')
 L.append('')
 for name in run['coverage']:
  L.append(f'### {name}');L.append('')
  rows_out=[]
  def cells(construct,role,blocks,rule,pearson,mae,exposed):
   rows_out.append([construct,role,blocks,rule,pearson,mae,'yes' if exposed else 'no'])
  for construct in CONSTRUCTS:
   byrule=lambda system,exposed:[r for r in records if r['system']==system
     and r['construct']==construct and r['potentially_optimistic']==exposed]
   clean=[r for r in byrule('S3',False)]
   if clean:
    r=max(clean,key=lambda r:(r['development_spearman'],-r['development_mae']))
    cells(construct,'clean reference, X6 rule on dev','+'.join(r['blocks']),str(r['rule']),
          cell(r['coverage'][name],'pearson'),cell(r['coverage'][name],'mae',3),False)
   for system in ('S1','S2'):
    chosen=[r for r in byrule(system,True) if r['rule']==PRESPECIFIED_RULE]
    for r in chosen:
     cells(construct,f'{system}, exposed, {PRESPECIFIED_RULE} rule as released','+'.join(r['blocks']),
           str(r['rule']),cell(r['coverage'][name],'pearson'),cell(r['coverage'][name],'mae',3),True)
   exposed3=byrule('S3',True)
   if exposed3:
    r=max(exposed3,key=lambda r:(r['development_spearman'],-r['development_mae']))
    cells(construct,'S3, exposed, same dev rule','+'.join(r['blocks']),str(r['rule']),
          cell(r['coverage'][name],'pearson'),cell(r['coverage'][name],'mae',3),True)
   seeds=[r for r in byrule('S4',True) if r.get('in_s1_pairing')]
   if seeds:
    pe=np.array([r['coverage'][name]['pearson'] for r in seeds],float)
    ma=np.array([r['coverage'][name]['mae'] for r in seeds],float)
    cells(construct,f'S4, exposed, S1 block set, {len(seeds)} seeds','+'.join(seeds[0]['blocks']),
          'mean over seeds',f'{pe.mean():.4f} ± {pe.std(ddof=1):.4f}',
          f'{ma.mean():.3f} ± {ma.std(ddof=1):.3f}',True)
   for exposed,label in ((False,'clean'),(True,'exposed')):
    pool=[r for r in records if r['construct']==construct and r['potentially_optimistic']==exposed
          and (r['system']!='S4' or r.get('in_s1_pairing'))]
    if pool:
     r=max(pool,key=lambda r:r['coverage'][name]['pearson'])
     cells(construct,f'descriptive, best {label} by test Pearson','+'.join(r['blocks']),
           f"{r['system']} {r['rule']}",cell(r['coverage'][name],'pearson'),
           cell(r['coverage'][name],'mae',3),exposed)
  table(L,rows_out,lambda r:r,
    ['construct','role','blocks','rule','Pearson [95% CI] or mean ± sd','MAE [95% CI] or mean ± sd','exposed'])
 L.append('## Published reference points');L.append('')
 L.append('Recorded in PROTOCOL.md before this module produced a number, so they')
 L.append('could not inform any choice here. They are not comparisons computed by')
 L.append('this module: the splits, the targets and the metric conventions of those')
 L.append('papers were not re-derived.');L.append('')
 for who,what in LITERATURE:L.append(f'- {who}: {what}')
 L.append('')
 L.append('Accuracy figures, and the numbers of Kim et al. 2022, are to be confirmed')
 L.append('against the primary sources at writing time.')
 L.append('')
 (path/'tables.md').write_text('\n'.join(L))
 return len(L)

def s1_block_sets():
 """The block set each released head uses, read from the checkpoints."""
 out={}
 for family,rule in RULES:
  for construct in CONSTRUCTS:
   meta=json.loads((ROOT/'models'/family/rule/'speechocean'/f'{construct}.json').read_text())
   blocks=tuple(meta['blocks'])
   if out.setdefault(construct,blocks)!=blocks:
    raise ValueError(f'The released heads disagree on the block set for {construct}')
 return out

def coverage_sets(test):
 """T-cov follows what the application accepts, which is the phone indicator.

 The release refuses a reference when either indicator is false, and
 text_to_phones inserts <UNKNOWN:word> for every word outside the dictionary,
 so incomplete text implies an incomplete phone reference. Reading the phone
 indicator therefore reproduces the application. It is read strictly: a missing
 field must fail rather than turn T-cov into T-all.
 """
 rows=test['rows']
 missing=[r['recording_id'] for r in rows if 'reference_complete' not in r or 'text_complete' not in r]
 if missing:raise ValueError(f'{len(missing)} rows carry no coverage indicator, first {missing[0]}')
 complete=np.array([bool(r['reference_complete']) for r in rows])
 # Incomplete text must imply an incomplete phone reference.
 broken=[r['recording_id'] for r in rows if not r['text_complete'] and r['reference_complete']]
 if broken:raise ValueError(f'text_complete false with reference_complete true, first {broken[0]}')
 divergent=[r['recording_id'] for r in rows if bool(r['text_complete'])!=bool(r['reference_complete'])]
 covered=int(complete.sum())
 # The protocol fixes 2,483 for the official test partition. A different count
 # means the coverage rule no longer describes the same utterances, so stop.
 if len(rows)==2500 and covered!=2483:
  raise ValueError(f'T-cov is {covered}, not the 2483 the protocol fixes; stop and record before any number')
 return ({'T-all':np.ones(len(rows),bool),'T-cov':complete},
         {'covered':covered,'rows':len(rows),'official_partition_check':len(rows)==2500,
          'rows_where_indicators_diverge':divergent})

def main():
 p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
 p.add_argument('--train',type=Path,required=True,help='feature cache manifest of the full train partition')
 p.add_argument('--test',type=Path,required=True,help='feature cache manifest of the official test partition')
 p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
 p.add_argument('--resamples',type=int,default=2000);p.add_argument('--seed',type=int,default=42)
 p.add_argument('--systems',default='S1,S2,S3,S4');p.add_argument('--threads',type=int,default=6)
 p.add_argument('--s4-epochs',type=int,default=S4_DEFAULTS['epochs'],help='only for smoke tests; the protocol fixes 300')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 default_output=Path(__file__).resolve().parent/'results'
 official=a.output.resolve()==default_output
 smoke=a.s4_epochs!=S4_DEFAULTS['epochs']
 if smoke and a.output.resolve()==default_output:
  raise ValueError('--s4-epochs other than the protocol value writes only outside the default results directory')
 wanted=[s.strip() for s in a.systems.split(',') if s.strip()]
 started=time.time()
 train=load_corpus(a.train);test=load_test(a.test)
 if {r['speaker_id'] for r in train['rows']}&{r['speaker_id'] for r in test['rows']}:raise ValueError('Speaker overlap between train and test')
 if {r['audio_sha256'] for r in train['rows']}&{r['audio_sha256'] for r in test['rows']}:raise ValueError('Audio overlap between train and test')
 if official:
  # The 2,483 check must not be skipped by a short manifest, so the official
  # output refuses anything but the two full partitions.
  if len(test['rows'])!=2500:raise ValueError(f"the official run needs the 2500 test rows, got {len(test['rows'])}")
  if len(train['rows'])!=2500:raise ValueError(f"the official run needs the 2500 train rows, got {len(train['rows'])}")
 sets,coverage_report=coverage_sets(test);speakers=[r['speaker_id'] for r in test['rows']]
 samples={};yranks={}
 for name,keep in sets.items():
  samples[name]=speaker_resamples(np.asarray(speakers)[keep],a.resamples,a.seed)
  yranks[name]={j:[rankdata(test['targets'][keep,j][idx]) for idx in samples[name]] for j in range(len(CONSTRUCTS))}
 config={**S4_DEFAULTS,'threads':a.threads,'epochs':a.s4_epochs}
 s1_sets=s1_block_sets()
 generators={'S1':lambda:s1(test),'S2':lambda:s2(train,test),'S3':lambda:s3(train,test),'S4':lambda:s4(train,test,config,s1_sets)}
 records=[];predictions={}
 for system in wanted:
  if system not in generators:raise ValueError(f'Unknown system {system}')
  for info,pred in generators[system]():
   j=list(CONSTRUCTS).index(info['construct'])
   key=f"{info['system']}|{info['rule']}|{'+'.join(info['blocks'])}|{info['construct']}"
   if key in predictions:raise ValueError(f'Duplicate system key {key}')
   predictions[key]=pred.astype(np.float32)
   row=dict(info);row['key']=key;row['potentially_optimistic']=exposed(info['blocks'])
   row['coverage']={}
   for name,keep in sets.items():
    y=test['targets'][keep,j];q=pred[keep]
    point=point_metrics(y,q)
    reference=float(pearsonr(y,q).statistic)
    if abs(point['pearson']-reference)>1e-10:raise ValueError('Pearson disagrees with scipy')
    reference=spearmanr(y,q).statistic
    if np.isfinite(reference) and abs(point['spearman']-float(reference))>1e-10:
     raise ValueError('Spearman disagrees with scipy')
    row['coverage'][name]={**point,'interval':intervals(y,q,samples[name],yranks[name][j]),
                           'speakers':len(set(np.asarray(speakers)[keep].tolist()))}
   records.append(row)
   print(f"{key}  T-all pearson {row['coverage']['T-all']['pearson']:.4f}  mae {row['coverage']['T-all']['mae']:.3f}",flush=True)
 np.savez_compressed(a.output/'predictions.npz',
   recording_ids=np.array([r['recording_id'] for r in test['rows']]),
   targets=test['targets'].astype(np.float32),
   reference_complete=np.array([bool(r['reference_complete']) for r in test['rows']]),
   text_complete=np.array([bool(r['text_complete']) for r in test['rows']]),
   speakers=np.array(speakers),**predictions)
 run={'schema_version':'x4-test-evaluation-v1','systems':wanted,'resamples':a.resamples,'seed':a.seed,
  'train_manifest_sha256':train['manifest_sha256'],'test_manifest_sha256':test['manifest_sha256'],
  'code_sha256':digest(__file__),'ridge_code_sha256':digest(ROOT/'reproduce/train.py'),
  'fusion_code_sha256':digest(ROOT/'speech/pronunciation/fusion.py'),
  's4_configuration':dict(config),
  'coverage':{k:int(v.sum()) for k,v in sets.items()},'coverage_check':coverage_report,
  'exposed_blocks':list(EXPOSED_BLOCKS),'smoke_test':bool(smoke),'seconds':round(time.time()-started,1),
  'predictions_sha256':digest(a.output/'predictions.npz')}
 save(a.output/'metrics.json',{'run':run,'records':records})
 lines=write_tables(records,run,a.output)
 print(json.dumps({**{k:run[k] for k in ('systems','coverage','seconds')},'records':len(records),'table_lines':lines}))
if __name__=='__main__':main()
