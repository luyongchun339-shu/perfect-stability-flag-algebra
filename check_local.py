#!/usr/bin/env python3
"""Exact local counting checks, using only the Python standard library.

Checks the candidate density, pair cancellation, all occupancy patterns,
polynomial coefficients and finite examples. The edit-distance lower bound
is a mathematical argument in MATHEMATICS.md, not a claim of this script.
"""
from __future__ import annotations
from fractions import Fraction as F
from itertools import combinations
from math import comb,factorial,prod
from pathlib import Path
import argparse,csv,json
from flag_model import *

def component_partition_fast(occ):
    x,y,a,b,c,u=occ
    if a and b and c:return None
    parts=[v for v in (x,y) if v]
    if c:parts.append(a+b+c)
    else:parts.extend(v for v in (a,b) if v)
    parts.extend([1]*u)
    return tuple(sorted(parts,reverse=True))

def component_partition_direct(occ):
    labels=[i for i,count in enumerate(occ) for _ in range(count)]
    adj=[set() for _ in labels]
    for i,j in combinations(range(len(labels)),2):
        a,b=labels[i],labels[j]
        if (a==b and a!=5) or {a,b} in ({2,4},{3,4}):
            adj[i].add(j);adj[j].add(i)
    unseen=set(range(len(labels)));parts=[]
    while unseen:
        first=min(unseen);component={first};todo=[first]
        while todo:
            i=todo.pop()
            for j in adj[i]-component:
                component.add(j);todo.append(j)
        if any(len(adj[i]&component)!=len(component)-1 for i in component):return None
        unseen-=component;parts.append(len(component))
    return tuple(sorted(parts,reverse=True))

def poly_value(coefficients,e):
    value=F(0)
    for c in reversed(coefficients):value=value*e+c
    return value

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',default='results/local');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    good=[];tested=0
    for occ in weak_compositions(ORDER,6):
        fast=component_partition_fast(occ);direct=component_partition_direct(occ)
        require(fast==direct,'Occupancy classifier mismatch '+str(occ));tested+=1
        if fast==TARGET:good.append(occ)
    coefficients=[F(0)]*(ORDER+1)
    for occ in good:
        x,y,a,b,c,u=occ;degree=a+b+c
        ways=factorial(ORDER)//prod(factorial(z) for z in occ)
        w=F(ways,5**(x+y)*3**degree)
        for j in range(u+1):coefficients[degree+j]+=w*comb(u,j)*F(3,5)**(u-j)*(-1)**j
    x,q=F(1,5),F(3,5)
    candidate=F(factorial(15),factorial(4)*factorial(2)*factorial(9))*2*x**6*q**9
    edge=F(factorial(13),factorial(4)*factorial(2)*factorial(7))*2*x**6*q**7
    nonedge=F(factorial(13),factorial(4)*factorial(9))*2*x**4*(q**9+9*x*q**8)
    require(candidate==BOUND==coefficients[0],'Candidate value mismatch')
    require(edge==nonedge==F(20,21)*BOUND,'Two-root cancellation failed')
    require(coefficients[1:3]==[0,0],'Low-order cancellation failed')
    require(coefficients[3]==-F(424702278,9765625),'Wrong cubic term')
    require(coefficients[4]==F(866918052,1953125),'Wrong quartic term')
    require(len(good)==90,'Wrong successful occupancy count')
    finite=[]
    for k in (2,3,5,10):
        n=15*k**4;sizes=(3*k**4,3*k**4,5*k**3,5*k**3,5*k**3,9*k**4-15*k**3)
        count=sum(prod(comb(s,t) if s>=t else 0 for s,t in zip(sizes,o)) for o in good)
        density=F(count,comb(n,15));limit=poly_value(coefficients,F(1,k))
        require(abs(density-limit)<=F(105,n),'Finite sampling comparison failed')
        delta=F(2*(5*k**3)**2,n*n)
        require(delta==F(2,9*k*k),'Missing normalization factor 2')
        finite.append(dict(k=k,n=n,density=str(density),graphon_density=str(limit),normalized_edit_distance=str(delta),collision_bound=str(F(105,n))))
    face_coeff=[F(0)]*3
    for power,coef,degree in [(0,2*x**6,9),(1,18*x**6,8),(2,2*x**4,9),(2,18*x**5,8)]:
        for j in range(min(degree,2-power)+1):
            face_coeff[power+j]+=75075*coef*comb(degree,j)*q**(degree-j)*(-1)**j
    require(face_coeff==[BOUND,F(0),F(0)],'Face reduction cancellation failed')
    report=dict(status='EXACT_LOCAL_CHECKS_PASSED',occupancies_tested=tested,successful_occupancies=len(good),
        candidate_density=str(candidate),root_edge=str(edge),root_nonedge=str(nonedge),
        coefficients_low_to_high=list(map(str,coefficients)),face_coefficients_0_to_2=list(map(str,face_coeff)),
        finite_examples=finite,edit_distance_scope='Checks arithmetic after the combinatorial lower bound in MATHEMATICS.md')
    dump_json(out/'local_report.json',report)
    with (out/'polynomial_coefficients.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['degree','exact_coefficient']);w.writerows(enumerate(map(str,coefficients)))
    dump_json(out/'successful_occupancies.json',[list(x) for x in good])
    print('LOCAL_EXACT_PASS',tested,'occupancies;',len(good),'successful; normalized edit distance 2/(9*k^2)',flush=True)

if __name__=='__main__':main()
