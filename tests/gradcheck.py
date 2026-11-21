"""Finite-difference check of MLP.backward (course Appendix A). Run: python tests/gradcheck.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlp import MLP, bce
rng=np.random.default_rng(0)
worst=0
for hidden in [(32,),(16,8)]:
    m=MLP(45,hidden,weight_decay=1e-4,seed=3)
    X=rng.normal(size=(64,45)); y=(rng.random(64)<0.3).astype(float)
    def J(params):
        m.set_params(params); p=m.predict_proba(X)
        return bce(p,y)+0.5*m.weight_decay*sum((w**2).sum() for w in m.W)
    P=[p.copy() for p in m.params]; m.set_params(P)
    p,c=m.forward(X); G=m.backward(p,y,c)
    for i,(par,g) in enumerate(zip(P,G)):
        num=np.zeros_like(par); it=np.nditer(par,flags=['multi_index'])
        for _ in it:
            ix=it.multi_index; h=1e-6
            Q=[q.copy() for q in P]; Q[i][ix]+=h; a=J(Q)
            Q=[q.copy() for q in P]; Q[i][ix]-=h; b=J(Q)
            num[ix]=(a-b)/(2*h)
        rel=np.abs(num-g).max()/max(1e-12,np.abs(num).max()+np.abs(g).max())
        worst=max(worst,rel)
    print(f'hidden={hidden}: max relative error {worst:.1e}')
assert worst < 1e-6, 'gradient check failed'
print('gradient check passed')
