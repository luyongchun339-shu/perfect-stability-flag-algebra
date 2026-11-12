#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,json,time
from itertools import permutations,combinations
from collections import Counter,defaultdict
from pathlib import Path
from flag_model import *
from verify_certificate import check_certificate

def brute(host,sigma,basis):
    labels=[j for j,s in enumerate(host) for _ in range(s)]
    n=len(labels);k=sum(sigma);size=(n-k)//2
    root_type=[j for j,s in enumerate(sigma) for _ in range(s)]
    index={flag:i for i,flag in enumerate(basis)};result=defaultdict(int)
    for roots in permutations(range(n),k):
        if any((labels[roots[i]]==labels[roots[j]])!=(root_type[i]==root_type[j]) for i in range(k) for j in range(i)):continue
        root_components=[];offset=0
        for s in sigma:root_components.append(labels[roots[offset]]);offset+=s
        def flag_id(side):
            counts=Counter(labels[i] for i in side)
            a=tuple(counts[c] for c in root_components)
            nu=tuple(sorted((v for c,v in counts.items() if c not in root_components),reverse=True))
            return index[a,nu]
        rest=[i for i in range(n) if i not in roots]
        for left in combinations(rest,size):
            right=[i for i in rest if i not in left]
            result[flag_id(left),flag_id(right)]+=1
    return [[i,j,v] for (i,j),v in sorted(result.items()) if i<=j]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--certificate',default='results/sharp_certificate.json.gz');ap.add_argument('--out',default='results/tests');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    cases=0
    for n in (5,7):
        for sigma in TYPES:
            basis=flags(n,sigma)
            for host in partitions(n):
                require(count_matrix(host,sigma,basis)==brute(host,sigma,basis),'Brute count mismatch')
                cases+=1
    model=build_model();cert=load_json(args.certificate)
    rejected=[]
    def check_bad(name,bad):
        try:check_certificate(bad,model,verbose=False)
        except (ValueError,KeyError,TypeError,ArithmeticError):rejected.append(name)
        else:raise ValueError('Corrupt certificate was accepted: '+name)
    bad=copy.deepcopy(cert);bad['bound']='0';check_bad('wrong_bound',bad)
    bad=copy.deepcopy(cert);bad['blocks'][0]['S'][0][0]=-abs(bad['blocks'][0]['S'][0][0])-1;check_bad('indefinite_factor',bad)
    bad=copy.deepcopy(cert);bad['residuals'][0]=str(Fraction(bad['residuals'][0])+1);check_bad('forged_residual',bad)
    bad=copy.deepcopy(cert);bad['blocks'][0]['flags'][0],bad['blocks'][0]['flags'][1]=bad['blocks'][0]['flags'][1],bad['blocks'][0]['flags'][0];check_bad('permuted_flags_without_matrices',bad)
    bad=copy.deepcopy(cert);bad['blocks'][0]['N_denominator']=0;check_bad('zero_denominator',bad)
    bad=copy.deepcopy(cert);bad['blocks'][0]['S'][0][0]=float(bad['blocks'][0]['S'][0][0]);check_bad('floating_point_certificate_entry',bad)
    k=3;n=15*k**4
    require(Fraction((5*k**3)**2,n*n)!=Fraction(2,9*k*k),'Normalization regression failed')
    report=dict(status='TESTS_PASSED',brute_labelled_host_cases=cases,corrupt_certificates_rejected=rejected,factor_two_regression='PASS',elapsed_seconds=time.monotonic()-start)
    dump_json(out/'test_report.json',report)
    print('TESTS_PASSED',cases,'brute cases;',len(rejected),'corrupt certificates rejected',flush=True)

if __name__=='__main__':main()
