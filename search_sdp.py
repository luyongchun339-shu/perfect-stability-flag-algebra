#!/usr/bin/env python3
"""Fresh SDP search by a log-determinant path-following method.

This file implements the numerical solver, rather than pretending to call an
unavailable solver. Only NumPy/SciPy linear algebra is used. No certificate,
stored solution, or previous run is read. A numerical success is NOT a proof.
Run recover_certificate.py and both exact verifiers afterwards.

Fix t=M and search for a positive margin on the analytically derived face.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, csv, hashlib, json, platform, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import linalg, special
import scipy, sympy
from flag_model import *
from face import build_face
from sdp_core import numerical_blocks,svec

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ewens_start(hosts,theta=2.0):
    n=sum(hosts[0])
    return np.array([np.exp(special.gammaln(n+1)+special.gammaln(theta)
        -special.gammaln(n+theta)+len(h)*np.log(theta)
        -sum(np.log(a) for a in h)
        -sum(special.gammaln(c+1) for c in Counter(h).values())) for h in hosts])

def search(model,out,max_stages=12,max_newton=80):
    start=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    face=build_face(model)
    dump_json(out/'face.json.gz',face)
    Bs,Os,Rs=numerical_blocks(model,face)
    dims=[B.shape[1] for B in Bs];h=len(model['partitions'])
    target=np.array([int(tuple(host)==TARGET) for host in model['partitions']],dtype=float)
    nonnegative=[i for i in range(h) if i not in face['zero_slacks']]
    zero=face['zero_slacks']
    L=np.hstack([np.array([svec(B[i]) for i in range(h)]) for B in Bs]+[np.eye(h)[:,nonnegative]])
    b=float(BOUND)-target
    a=np.array([sum(np.trace(B[i]) for B in Bs)+(i in nonnegative) for i in range(h)])
    eigen,U=linalg.eigh(L@L.T)
    U=U[:,eigen>eigen[-1]*1e-12]
    Z=U@linalg.null_space((a@U)[None,:])
    y0=U@(U.T@ewens_start(model['partitions']));y0/=a@y0
    B0=[np.einsum('i,ijk->jk',y0,B) for B in Bs]
    C=[np.einsum('ji,jkl->ikl',Z,B) for B in Bs]
    g0=Z.T@b;Zpos=Z[nonnegative];yp0=y0[nonnegative]
    require(min(yp0)>0,'Initial scalar cone is not strictly feasible')
    for matrix in B0: linalg.cholesky(matrix,lower=True)
    input_arrays=dict(objective=b,normalization=a,nonnegative=np.array(nonnegative),initial_y=y0)
    for i,B in enumerate(Bs): input_arrays[f'B{i}']=B;input_arrays[f'O{i}']=Os[i];input_arrays[f'R{i}']=Rs[i]
    np.savez_compressed(out/'sharp_inputs.npz',**input_arrays)
    print('SEARCH_START','sharp','dual_dimension',Z.shape[1],'block_dimensions',dims,'warm_start',False,flush=True)

    def evaluate(z,mu,derivatives=True):
        yp=yp0+Zpos@z
        if min(yp)<=0: return None
        matrices=[P+np.einsum('i,ijk->jk',z,Q) for P,Q in zip(B0,C)]
        inverses=[];logdet=0.0
        try:
            for matrix in matrices:
                chol=linalg.cholesky(matrix,lower=True,check_finite=False)
                logdet+=2*np.log(np.diag(chol)).sum()
                inverses.append(linalg.cho_solve((chol,True),np.eye(len(matrix)),check_finite=False))
        except linalg.LinAlgError: return None
        value=b@y0+g0@z-mu*(logdet+np.log(yp).sum())
        if not derivatives: return value
        gradient=g0-mu*Zpos.T@(1/yp)
        H=(Zpos.T/(yp**2))@Zpos
        for inv,Q in zip(inverses,C):
            gradient-=mu*np.einsum('ab,iab->i',inv,Q)
            products=Q@inv
            H+=products.reshape(len(z),-1)@products.transpose(0,2,1).reshape(len(z),-1).T
        return value,gradient,mu*(H+H.T)/2,inverses,yp

    z=np.zeros(Z.shape[1]);mu=1e-3
    value,gradient,H,*_=evaluate(z,mu)
    chol=linalg.cholesky(H,lower=True)
    change=linalg.solve_triangular(chol.T,np.eye(len(z)),lower=False)
    Z=Z@change;C=[np.einsum('ji,jkl->ikl',change,Q) for Q in C]
    g0=Z.T@b;Zpos=Z[nonnegative]
    history=[];stages=[];accepted=False
    total_dimension=sum(dims)+len(nonnegative)
    csvfile=(out/'sharp_iterations.csv').open('w',newline='',encoding='utf-8')
    writer=csv.writer(csvfile)
    writer.writerow(['stage','newton_step','mu','barrier_value','linear_objective','newton_decrement','gradient_norm','step_length','elapsed_seconds'])
    try:
        for stage in range(max_stages):
            for it in range(max_newton):
                value,gradient,H,inverses,yp=evaluate(z,mu)
                diagonal=np.sqrt(np.maximum(np.diag(H),1e-300))
                scaled=H/diagonal[:,None]/diagonal[None,:]
                try:
                    step_vector=linalg.solve(scaled,-gradient/diagonal,assume_a='pos')/diagonal
                except linalg.LinAlgError:
                    step_vector=linalg.lstsq(H,-gradient,cond=1e-13)[0]
                decrement=float(-gradient@step_vector)
                require(np.isfinite(decrement),'Nonfinite Newton decrement')
                step=0.0
                if decrement>=mu*1e-12:
                    step=1.0
                    for line in range(70):
                        candidate=evaluate(z+step*step_vector,mu,False)
                        if candidate is not None and candidate<=value+0.01*step*(gradient@step_vector)+1e-13*max(1.0,abs(value)):
                            break
                        step*=0.5
                    else: raise RuntimeError('Newton line search failed')
                row=[stage,it,mu,float(value),float(b@y0+g0@z),decrement,float(np.linalg.norm(gradient)),step,time.monotonic()-start]
                writer.writerow(row);csvfile.flush();history.append(row)
                if it%5==0: print('NEWTON',*row,flush=True)
                if step==0: break
                z+=step*step_vector
            value,gradient,H,inverses,yp=evaluate(z,mu)
            y=y0+Z@z
            delta=float(b@y-mu*total_dimension)
            X=[mu*inv+delta*np.eye(d) for inv,d in zip(inverses,dims)]
            bound=float(BOUND)
            residual=np.array([bound-target[j]-sum(np.sum(x*B[j]) for x,B in zip(X,Bs)) for j in range(h)])
            eig=[float(linalg.eigvalsh(x)[0]) for x in X]
            equality_error=float(max(abs(residual[zero]))) if zero else 0.0
            statistic=dict(stage=stage,mu=mu,barrier_gap=mu*total_dimension,bound=bound,
                           target_dual_coordinate=float(y@target),
                           margin=delta,
                           equality_error=equality_error,min_positive_slack=float(min(residual[nonnegative])),
                           reduced_min_eigenvalues=eig,elapsed_seconds=time.monotonic()-start)
            stages.append(statistic);print('STAGE',json.dumps(statistic),flush=True)
            arrays=dict(dual_variables=y,residuals=residual,mu=mu,delta=delta,bound=bound)
            for i,x in enumerate(X):arrays[f'X{i}']=x;arrays[f'O{i}']=Os[i];arrays[f'R{i}']=Rs[i]
            np.savez_compressed(out/'sharp_checkpoint.npz',**arrays)
            accepted=min(eig)>1e-8 and min(residual[nonnegative])>1e-8 and equality_error<1e-8
            if accepted:
                np.savez_compressed(out/'numerical_sharp.npz',**arrays)
                break
            mu*=0.2
    finally: csvfile.close()
    report=dict(status='NUMERICAL_CANDIDATE' if accepted else 'NOT_CONVERGED',mode='sharp',
        algorithm='log-determinant path-following with damped Newton steps',
        numerical_only=True,old_certificate_read=False,warm_start=False,
        initial_distribution='Ewens theta=2',block_dimensions=dims,
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in ['flag_model.py','face.py','sdp_core.py','search_sdp.py']},
        input_sha256=sha(out/'sharp_inputs.npz'),stages=stages,newton_steps=len(history),
        numpy=np.__version__,scipy=scipy.__version__,sympy=sympy.__version__,python=sys.version,
        platform=platform.platform(),timestamp_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-start)
    dump_json(out/'sharp_report.json',report)
    print('SEARCH_END',report['status'],'elapsed_seconds',report['elapsed_seconds'],flush=True)
    require(accepted,'No acceptable numerical candidate was found; no exact claim is made.')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',default='results/model.json.gz');ap.add_argument('--out',default='results')
    ap.add_argument('--mode',choices=['sharp'],default='sharp',help='Only the sharp certificate search is supported')
    ap.add_argument('--max-stages',type=int,default=12);ap.add_argument('--max-newton',type=int,default=80)
    args=ap.parse_args()
    search(load_json(args.model),args.out,args.max_stages,args.max_newton)
