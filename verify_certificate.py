#!/usr/bin/env python3
"""Exact certificate verification, Python standard library only.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from fractions import Fraction
from pathlib import Path
from flag_model import ORDER,TARGET,TYPES,BOUND,build_model,load_json,dump_json,require

def integer_matrix(matrix,rows,cols,name):
    require(isinstance(matrix,list) and len(matrix)==rows,name+' row count')
    require(all(isinstance(row,list) and len(row)==cols for row in matrix),name+' column count')
    require(all(type(x) is int for row in matrix for x in row),name+' must contain exact integers')
    return matrix

def congruence(N,S):
    sparse=[[(j,v) for j,v in enumerate(row) if v] for row in N]
    size=len(S)
    middle=[[sum(v*S[j][k] for j,v in row) for k in range(size)] for row in sparse]
    ans=[[0]*len(N) for _ in N]
    for i in range(len(N)):
        for k in range(i,len(N)):
            value=sum(middle[i][j]*v for j,v in sparse[k])
            ans[i][k]=ans[k][i]=value
    return ans

def check_certificate(certificate,model,verbose=True):
    require(certificate.get('schema')=='cluster-flag-factor-certificate-v1','Wrong certificate schema')
    require(certificate.get('order')==ORDER,'Wrong graph order')
    require(tuple(certificate.get('target_partition',()))==TARGET,'Wrong target')
    require(Fraction(certificate['bound'])==BOUND,'Wrong sharp bound')
    recs=certificate['blocks']
    require(len(recs)==4 and [tuple(x['type']) for x in recs]==list(TYPES),'Wrong type list or order')
    matrices=[];denominators=[];psd=[]
    for block,rec in zip(model['blocks'],recs):
        require(rec.get('flags')==block['flags'],'Wrong flag basis or order')
        size=len(block['flags']);rank=len(rec['S'])
        require(0<rank<=size,'Invalid factor dimension')
        N=integer_matrix(rec['N'],size,rank,'N')
        S=integer_matrix(rec['S'],rank,rank,'S')
        B=integer_matrix(rec['B'],rank,rank,'B')
        u=rec['N_denominator'];v=rec['S_denominator']
        require(type(u) is int and type(v) is int and u>0 and v>0,'Invalid denominator')
        require(all(S[i][j]==S[j][i] for i in range(rank) for j in range(i)),'S is not symmetric')
        D=congruence(B,S)
        margins=[D[i][i]-sum(abs(D[i][j]) for j in range(rank) if j!=i) for i in range(rank)]
        require(min(margins)>0,'Exact positive diagonal dominance failed')
        if 'diagonal_dominance_margin_min' in rec:
            require(int(rec['diagonal_dominance_margin_min'])==min(margins),'Incorrect stored PSD margin')
        Q=congruence(N,S);den=u*u*v
        matrices.append(Q);denominators.append(den)
        psd.append(dict(type=rec['type'],dimension=size,factor_dimension=rank,margin=str(min(margins))))
        if verbose:print('PSD_EXACT_PASS',rec['type'],'dimension',size,'factor',rank,flush=True)
    rows=[]
    for li,host in enumerate(model['partitions']):
        lhs=BOUND-int(tuple(host)==TARGET);total=Fraction(0)
        for b,Q,d in zip(model['blocks'],matrices,denominators):
            num=sum((1 if i==j else 2)*w*Q[i][j] for i,j,w in b['matrices_upper'][li])
            total+=Fraction(num,b['denominator']*d)
        residual=lhs-total
        require(residual>=0,'Negative exact residual for '+str(host))
        rows.append((host,lhs,total,residual))
    require(len(certificate.get('residuals',[]))==len(rows),'Wrong residual list length')
    for expected,row in zip(certificate['residuals'],rows):
        require(Fraction(expected)==row[3],'Stored residual does not match exact recomputation')
    return psd,rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--certificate',default='results/sharp_certificate.json.gz')
    ap.add_argument('--out',default='results/verified')
    args=ap.parse_args();start=time.monotonic();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    model=build_model();cert=load_json(args.certificate)
    psd,rows=check_certificate(cert,model)
    dump_json(out/'recomputed_model.json.gz',model)
    with (out/'residuals.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['partition','bound_minus_indicator','matrix_contribution','residual'])
        for host,lhs,rhs,res in rows:w.writerow([' '.join(map(str,host)),str(lhs),str(rhs),str(res)])
    positive=[r[3] for r in rows if r[3]>0]
    report=dict(status='VERIFIED',bound=str(BOUND),identities=len(rows),matrices=4*len(rows),
        zero_residuals=sum(r[3]==0 for r in rows),positive_residuals=len(positive),negative_residuals=0,
        min_positive_residual=str(min(positive)),psd=psd,arithmetic='Exact integers and fractions.Fraction',
        certificate_sha256=hashlib.sha256(Path(args.certificate).read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-start)
    dump_json(out/'report.json',report)
    print('GLOBAL_EXACT_VERIFICATION',report['status'],flush=True)
    print('BOUND',str(BOUND),'RESIDUALS',report['zero_residuals'],report['positive_residuals'],0,flush=True)
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,KeyError,TypeError,OSError,ArithmeticError) as exc:
        print('VERIFICATION_FAILED',str(exc),file=sys.stderr);sys.exit(1)
