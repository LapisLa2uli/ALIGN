"""Small, independently saved decoder attention adapters; base remains frozen."""
import contextlib
import hashlib
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self,base,rank=4,alpha=8,dropout=.05):
        super().__init__();self.base=base;self.scale=alpha/rank;self.enabled=True
        self.a=nn.Linear(base.in_features,rank,bias=False,device=base.weight.device,dtype=base.weight.dtype)
        self.b=nn.Linear(rank,base.out_features,bias=False,device=base.weight.device,dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.a.weight,a=5**.5);nn.init.zeros_(self.b.weight)
        self.dropout=nn.Dropout(dropout)
    @property
    def weight(self):return self.base.weight
    def forward(self,x):
        y=self.base(x)
        return y+self.scale*self.b(self.a(self.dropout(x))) if self.enabled else y

def install(model):
    model.requires_grad_(False);names=[]
    for name,module in list(model.named_modules()):
        if name.startswith('decoder.block.') and name.rsplit('.',1)[-1] in ('q','v') and isinstance(module,nn.Linear):
            parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,LoRALinear(module));names.append(name)
    assert names
    return names

def mode(model,training):
    model.eval()
    for module in model.modules():
        if isinstance(module,LoRALinear):module.dropout.train(training)

@contextlib.contextmanager
def disabled(model):
    modules=[m for m in model.modules() if isinstance(m,LoRALinear)]
    for m in modules:m.enabled=False
    try:yield
    finally:
        for m in modules:m.enabled=True

def state(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}

def load(model,values):
    params=dict(model.named_parameters());assert set(values)=={n for n,p in params.items() if p.requires_grad}
    with torch.no_grad():
        for n,v in values.items():params[n].copy_(v)

def frozen_hash(model):
    digest=hashlib.sha256()
    for n,p in model.named_parameters():
        if not p.requires_grad:digest.update(n.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
