#!/usr/bin/env python3
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import sys, time

from flag_model import *
from sdp_core import dense_counts,svec,smat
from scipy import linalg
import numpy as np
import sympy as sp
from functools import reduce
from math import gcd,lcm
F=Fraction
import argparse, hashlib
ap=argparse.ArgumentParser(description='Recover an exact rational certificate from a fresh numerical SDP solution.')
ap.add_argument('--out',default='results')
ap.add_argument('--digits',type=int,default=12)
args=ap.parse_args()
out=Path(args.out)
model=load_json(out/'model.json.gz');face=load_json(out/'face.json.gz');num=np.load(out/'numerical_sharp.npz')
zeros=face['zero_slacks'];dims=[];blocks=[];Hs=[];Sf=[];den=[];meta=[]
for i,(b,rec) in enumerate(zip(model['blocks'],face['blocks'])):
 N=np.array(rec['N'],dtype=np.int64);d=N.shape[1];dims.append(d)
 u=rec['N_denominator'];db=b['denominator'];den.append(db*u*u)
 ii,jj=np.triu_indices(d);mult=np.where(ii==jj,1,2)
 maxn=int(abs(N).max())
 require(maxn*maxn*db*2 < 2**63,'int64 bound insufficient')
 H=[]
 for j in zeros:
  C=dense_counts(b,j)
  Z=N.T@C@N
  H.append(Z[ii,jj]*mult)
 H=np.array(H,dtype=np.int64);Hs.append(H)
 R=num[f'R{i}'];X=num[f'X{i}']
 Ri=linalg.solve_triangular(R,np.eye(d),lower=False)
 S=Ri@X@Ri.T;S=(S+S.T)/2;Sf.append(S[ii,jj])
 meta.extend((i,int(a),int(c)) for a,c in zip(ii,jj))
 print('RECOVER_BLOCK',i,'d',d,'float min',linalg.eigvalsh(S)[0],flush=True)
Hf=np.hstack([H/d for H,d in zip(Hs,den)])
b=np.array([float(BOUND)-int(tuple(model['partitions'][j])==TARGET) for j in zeros])
xs=np.concatenate(Sf)
print('INITIAL_MAX_EQUALITY_ERROR',abs(Hf@xs-b).max(),flush=True)
_,R,perm=linalg.qr(Hf.T,pivoting=True,mode='economic')
rank=int(sum(abs(np.diag(R))>abs(R[0,0])*1e-10));rows=perm[:rank]
_,_,columns=linalg.qr(Hf[rows],pivoting=True,mode='economic');cols=columns[:rank]
print('LINEAR_RANK',rank,'pivotcond',np.linalg.cond(Hf[rows][:,cols]),flush=True)
xs[cols]+=linalg.solve(Hf[rows][:,cols],b[rows]-Hf[rows]@xs)
D=10**args.digits
z=np.array([int(round(x*D)) for x in xs],dtype=object)
cs=np.cumsum([0]+[H.shape[1] for H in Hs])
err=[BOUND-int(tuple(model['partitions'][j])==TARGET) for j in zeros]
for H,d,lo,hi in zip(Hs,den,cs[:-1],cs[1:]):
 vals=H.astype(object)@z[lo:hi]
 for j,v in enumerate(vals):err[j]-=F(int(v),d*D)
common=lcm(*den)
P=[]
for row in rows:
 line=[]
 for col in cols:
  bi=next(i for i in range(4) if cs[i]<=col<cs[i+1]);local=col-cs[bi]
  line.append(int(Hs[bi][row,local])*(common//den[bi]))
 P.append(line)
rhs=[sp.Rational((err[j]*common).numerator,(err[j]*common).denominator) for j in rows]
for i,line in enumerate(P):
 g=gcd(*line)
 if g:
  P[i]=[v//g for v in line];rhs[i]/=g
print('EXACT_LINEAR_SOLVE_START',flush=True);t=time.monotonic()
P=sp.Matrix(P);iv=P.inv(method='DM');corr=iv*sp.Matrix(rhs)
print('EXACT_LINEAR_SOLVE_SECONDS',time.monotonic()-t,flush=True)
sol=[F(int(v),D) for v in z]
for k,col in enumerate(cols):sol[col]+=F(int(corr[k].p),int(corr[k].q))
print('MAX_RATIONAL_CORRECTION',max(abs(float(c)) for c in corr),flush=True)
cert=dict(schema='cluster-flag-factor-certificate-v1',order=15,target_partition=list(TARGET),bound=str(BOUND),blocks=[],provenance='Fresh numerical log-det SDP search and exact rational recovery. No historical certificate was read.')
qnums=[];qdens=[]
for bi,(b,rec,d,lo,hi) in enumerate(zip(model['blocks'],face['blocks'],dims,cs[:-1],cs[1:])):
 sf=sol[lo:hi];v=lcm(*(x.denominator for x in sf));ii,jj=np.triu_indices(d)
 Sint=np.zeros((d,d),dtype=object)
 for a,c,x in zip(ii,jj,sf):Sint[a,c]=Sint[c,a]=x.numerator*(v//x.denominator)
 Sd=np.array([[float(F(int(x),v)) for x in row] for row in Sint])
 print('EXACT_S',bi,'denominator digits',len(str(v)),'minfloat',linalg.eigvalsh(Sd)[0],flush=True)
 L=linalg.cholesky(Sd,lower=True)
 Li=linalg.solve_triangular(L,np.eye(d),lower=True)
 for scale in [10**5,10**7,10**9,10**11]:
  B=np.array([[int(round(scale*x)) for x in row] for row in Li],dtype=object)
  DD=B@Sint@B.T
  margins=[int(DD[i,i]-sum(abs(DD[i,j]) for j in range(d) if j!=i)) for i in range(d)]
  if min(margins)>0:break
 else:raise ValueError('Cannot certify diagonal dominance')
 N=np.array(rec['N'],dtype=object);QN=N@Sint@N.T;QD=rec['N_denominator']**2*v
 qnums.append(QN);qdens.append(QD)
 cert['blocks'].append(dict(type=b['type'],flags=b['flags'],N=N.tolist(),N_denominator=rec['N_denominator'],S=Sint.tolist(),S_denominator=v,B=B.tolist(),diagonal_dominance_margin_min=str(min(margins))))
 print('PSD_EXACT_PASS',b['type'],'witness scale',scale,flush=True)
res=[]
for j,host in enumerate(model['partitions']):
 val=BOUND-int(tuple(host)==TARGET)
 for b,QN,QD in zip(model['blocks'],qnums,qdens):
  vv=sum((1 if i==k else 2)*w*QN[i,k] for i,k,w in b['matrices_upper'][j])
  val-=F(int(vv),b['denominator']*QD)
 require(val>=0,f'NEGATIVE residual {host}: {float(val)}')
 res.append(val)
cert['residuals']=[str(v) for v in res]
print('RESIDUALS',len(res),'zero',sum(v==0 for v in res),'positive',sum(v>0 for v in res),'minimum positive',float(min(v for v in res if v>0)),flush=True)
dump_json(out/'sharp_certificate.json.gz',cert)
report=dict(numerical_input_sha256=hashlib.sha256((out/'numerical_sharp.npz').read_bytes()).hexdigest(),certificate_sha256=hashlib.sha256((out/'sharp_certificate.json.gz').read_bytes()).hexdigest(),status='EXACT_RECOVERY_VERIFIED',zero=sum(v==0 for v in res),positive=sum(v>0 for v in res),min_positive=str(min(v for v in res if v>0)),pivot_rows=[int(zeros[j]) for j in rows],pivot_entries=[list(meta[j]) for j in cols],rounding_denominator=D)
dump_json(out/'recovery_report.json',report)
