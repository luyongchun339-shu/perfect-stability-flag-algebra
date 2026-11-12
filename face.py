from __future__ import annotations
from itertools import product
from math import factorial, lcm
import sympy as sp
from flag_model import *

def root_vectors(sigma,basis):
    m=len(sigma);b=(ORDER-sum(sigma))//2
    index={f:i for i,f in enumerate(basis)}
    rows=set()
    for where in product(range(4),repeat=m):
        if any(s!=1 and c==3 for s,c in zip(sigma,where)): continue
        cl=[c for c in where if c!=3]
        if len(cl)!=len(set(cl)): continue
        exponent=sum(s for s,c in zip(sigma,where) if c==2)
        if exponent>2: continue
        row=[0]*len(basis)
        for a in range(b+1):
            for d in range(b-a+1):
                e=b-a-d
                add=[a,d]
                extras=tuple(add[c] if c<2 else 0 for c in where)
                other=[add[c] for c in (0,1) if c not in where and add[c]]+[1]*e
                nu=tuple(sorted(other,reverse=True))
                w=factorial(b)//(factorial(a)*factorial(d)*factorial(e))*3**e
                row[index[extras,nu]]+=w
        rows.add(tuple(row))
    return sorted(rows)

def face_basis(sigma,basis):
    K=sp.Matrix(root_vectors(sigma,basis))
    import numpy as np
    from scipy import linalg
    rank=K.rank()
    _,_,order=linalg.qr(np.array(K.tolist(),dtype=float),pivoting=True,mode='economic')
    pivots=[int(j) for j in order[:rank]]
    free=[j for j in range(len(basis)) if j not in pivots]
    perm=pivots+free
    R,pv=K[:,perm].rref()
    require(tuple(pv)==tuple(range(rank)), 'Pivot columns are not exactly independent')
    N=sp.zeros(len(basis),len(free))
    for c,j in enumerate(free):
        N[j,c]=1
        for i,p in enumerate(pivots): N[p,c]=-R[i,rank+c]
    den=lcm(*(int(x.q) for x in N))
    Ni=[[int(den*N[i,j]) for j in range(N.cols)] for i in range(N.rows)]
    require(K*sp.Matrix(Ni)==sp.zeros(K.rows,N.cols),'Kernel basis error')
    return dict(type=list(sigma),kernel=[list(r) for r in root_vectors(sigma,basis)],
                pivots=list(pivots),free_columns=free,N=Ni,N_denominator=den)

def forced_zero(host):
    parts=[a for a in host if a>=2]
    return len(parts)<=2 or (len(parts)==3 and min(parts)==2)

def build_face(model):
    blocks=[]
    for block in model['blocks']:
        sig=tuple(block['type']);fs=[(tuple(a),tuple(nu)) for a,nu in block['flags']]
        rec=face_basis(sig,fs);blocks.append(rec)
        print('FACE',sig,'kernel',len(rec['pivots']),'dimension',len(rec['free_columns']),'denominator',rec['N_denominator'],flush=True)
    zeros=[i for i,h in enumerate(model['partitions']) if forced_zero(h)]
    print('FORCED_ZERO_SLACKS',len(zeros),flush=True)
    return dict(schema='cluster-flag-face-v1',blocks=blocks,zero_slacks=zeros)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='results/model.json.gz');ap.add_argument('--out',default='results/face.json.gz')
    args=ap.parse_args();dump_json(args.out,build_face(load_json(args.model)))
