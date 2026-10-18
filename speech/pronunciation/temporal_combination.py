"""Training-only convex blending of independently out-of-fold predictions."""
import numpy as np
from .development import metrics

def select_weights(targets,ridge,temporal,grid):
    arrays=[np.asarray(v,dtype=float) for v in (targets,ridge,temporal)]
    if arrays[0].ndim!=2 or any(v.shape!=arrays[0].shape or not np.isfinite(v).all() for v in arrays):raise ValueError('Expected matching finite target/prediction matrices')
    grid=np.asarray(grid,dtype=float)
    if grid.ndim!=1 or not len(grid) or not np.isfinite(grid).all() or ((grid<0)|(grid>1)).any() or len(set(grid))!=len(grid):raise ValueError('Invalid convex weight grid')
    y,r,t=arrays;weights=[];diagnostics=[]
    for j in range(y.shape[1]):
        candidates=[]
        for weight in grid:
            value=metrics(y[:,j],(1-weight)*r[:,j]+weight*t[:,j])
            if value['spearman'] is None:raise ValueError('Undefined blend-selection correlation')
            candidates.append(dict(temporal_weight=float(weight),metrics=value))
        best=max(candidates,key=lambda v:(v['metrics']['spearman'],-v['metrics']['mae'],-v['temporal_weight']))
        weights.append(best['temporal_weight']);diagnostics.append(candidates)
    return np.asarray(weights),diagnostics

def combine(ridge,temporal,weights):
    r,t,w=map(np.asarray,(ridge,temporal,weights))
    if r.ndim!=2 or r.shape!=t.shape or w.shape!=(r.shape[1],) or any(not np.isfinite(x).all() for x in (r,t,w)) or ((w<0)|(w>1)).any():raise ValueError('Invalid blend inputs')
    return r*(1-w)+t*w
