"""Separate supervision heads, shared frozen-feature network; no encoder training."""
import copy
import numpy as np
import torch
from torch import nn
from .fusion import FusionNetwork,DIMENSIONS
from .scoring import masked_loss


class SourceNetwork(FusionNetwork):
    def __init__(self,width,hidden=128,dropout=.2):
        super().__init__(width,hidden,dropout)
        self.azure_heads=nn.ModuleDict({d:nn.Linear(hidden,1) for d in DIMENSIONS})

    def forward(self,x):
        h=self.shared(x)
        return {source:{d:head(h).squeeze(-1) for d,head in heads.items()}
                for source,heads in [('speechocean',self.heads),('azure',self.azure_heads)]}

    def human_state(self):
        return {k:v for k,v in self.state_dict().items() if not k.startswith('azure_heads.')}


def scores(model,x,source):
    model.eval()
    with torch.no_grad():
        result=model(torch.as_tensor(x,dtype=torch.float32))[source]
        return np.stack([result[d].numpy() for d in DIMENSIONS],axis=1)


def errors(y,pred):
    result={}
    for j,d in enumerate(DIMENSIONS):
        mask=np.isfinite(y[:,j]);result[d]=None if not mask.any() else float(np.abs(y[mask,j]-pred[mask,j]).mean())
    return result


def fit_stage(model,x,y,train,valid,source,config,seed):
    """Targets normalized by their source scale; every row has >=1 observed label."""
    x=np.asarray(x,dtype=np.float32);y=np.asarray(y,dtype=np.float32)
    train=np.asarray(train,dtype=bool);valid=np.asarray(valid,dtype=bool)
    if x.ndim!=2 or y.shape!=(len(x),3) or train.shape!=(len(x),) or valid.shape!=(len(x),):raise ValueError('Training shapes disagree')
    if not np.isfinite(x).all() or np.isinf(y).any() or not np.isfinite(y).any(1).all():raise ValueError('Invalid features/labels')
    if ((y[np.isfinite(y)]<0)|(y[np.isfinite(y)]>1)).any():raise ValueError('Labels must be normalized to 0..1')
    if (train&valid).any() or not (train|valid).all() or not train.any() or not valid.any():raise ValueError('Invalid split')
    for j in range(3):
        if np.isfinite(y[train,j]).any()!=np.isfinite(y[valid,j]).any():raise ValueError('Label dimension must be represented in both splits or neither')
    torch.manual_seed(seed)
    active=list(model.shared.parameters())+list((model.azure_heads if source=='azure' else model.heads).parameters())
    optimizer=torch.optim.AdamW(active,lr=config['learning_rate'],weight_decay=config['weight_decay'])
    xt=torch.from_numpy(x[train]);yt=torch.from_numpy(y[train]);best=float('inf');history=[]
    for epoch in range(1,config['epochs']+1):
        model.train();total=0
        for ids in torch.randperm(len(xt)).split(config['batch_size']):
            model.zero_grad(set_to_none=True)
            loss=masked_loss(model(xt[ids]),{source:{d:yt[ids,j] for j,d in enumerate(DIMENSIONS)}})
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(ids)
        tr=errors(y[train],scores(model,x[train],source));va=errors(y[valid],scores(model,x[valid],source))
        # Azure selection uses available dimensions; human endpoint remains accuracy.
        objective=float(np.mean([v for v in va.values() if v is not None])) if source=='azure' else va['accuracy']
        history.append(dict(epoch=epoch,training_loss=total/len(xt),train_mae=tr,validation_mae=va))
        if objective<best:
            best=objective;bestepoch=epoch;state=copy.deepcopy(model.state_dict());opt=copy.deepcopy(optimizer.state_dict())
    model.load_state_dict(state);model.eval()
    return dict(best_epoch=bestepoch,history=history,objective=best),opt
