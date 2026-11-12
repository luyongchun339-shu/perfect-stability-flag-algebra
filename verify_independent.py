#!/usr/bin/env python3
import csv
from fractions import Fraction
from math import comb, factorial
from pathlib import Path
import gzip
import hashlib
import json
import sys
import time

N=15
TARGET=(4,2)+(1,)*9
M=Fraction(118216098,1220703125)
TYPES=((1,),(3,),(2,1),(1,1,1))


def check(test,message):
    if not test:raise ValueError(message)


def all_partitions(n,minimum=1):
    if n==0:
        yield ()
        return
    for first in range(minimum,n+1):
        for tail in all_partitions(n-first,first):
            yield (first,)+tail


def transpose(a):return list(map(list,zip(*a)))


def multiply(a,b):
    cols=transpose(b)
    out=[]
    for row in a:
        sparse=[(i,x) for i,x in enumerate(row) if x]
        out.append([sum(x*column[i] for i,x in sparse) for column in cols])
    return out


def matrix(a,r,c,name):
    check(isinstance(a,list) and len(a)==r,name+' rows')
    check(all(isinstance(row,list) and len(row)==c for row in a),name+' columns')
    check(all(type(x) is int for row in a for x in row),name+' noninteger')
    return a


def build_flag_set(m,r):
    ans=set()
    def rec(extra,left):
        if len(extra)==r:
            for p in all_partitions(left):ans.add((extra,tuple(reversed(p))))
        else:
            for a in range(left+1):rec(extra+(a,),left-a)
    rec((),m)
    return ans


def prepare_block(record):
    typ=tuple(record['type']);k=sum(typ);m=(N-k)//2
    fs=[(tuple(a),tuple(b)) for a,b in record['flags']]
    check(len(set(fs))==len(fs) and set(fs)==build_flag_set(m,len(typ)),
          'Incomplete, duplicate or invalid flag basis.')
    d=len(fs);q=len(record['S'])
    S=matrix(record['S'],q,q,'S')
    B=matrix(record['B'],q,q,'B')
    R=matrix(record['N'],d,q,'N')
    dn,ds=record['N_denominator'],record['S_denominator']
    check(type(dn) is int and type(ds) is int and dn>0 and ds>0,'denominator')
    check(S==transpose(S),'S not symmetric')
    check(all(B[i][i]!=0 and all(B[i][j]==0 for j in range(i+1,q))
              for i in range(q)),'B is not nonsingular lower triangular')
    T=multiply(multiply(B,S),transpose(B))
    check(T==transpose(T),'Congruence is not symmetric')
    margins=[T[i][i]-sum(abs(T[i][j]) for j in range(q) if j!=i) for i in range(q)]
    check(min(margins)>0,'Integer diagonal dominance failed')
    Q=multiply(multiply(R,S),transpose(R))
    check(Q==transpose(Q),'Q not symmetric')
    denominator=dn*dn*ds
    print('INDEPENDENT PSD',typ,'PASS',flush=True)
    return {'type':typ,'index':{f:i for i,f in enumerate(fs)},
            'Q':Q,'denominator':denominator,'m':m}


def add_part(p,a):
    return tuple(sorted(p+(a,),reverse=True)) if a else p


def joint_counts(host,block):
    typ=block['type'];r=len(typ);k=sum(typ);m=block['m']
    root_sizes=[sum(typ[j] for j in range(r) if mask&(1<<j)) for mask in range(1<<r)]
    empty=(-1,)*r
    states={(0,empty,empty,(),(),0):1}
    seen=0
    for s in host:
        new={}
        for (mask,le,re,lp,rp,L),count in states.items():
            R=seen-root_sizes[mask]-L
            for a in range(max(0,R+s-m),min(s,m-L)+1):
                key=(mask,le,re,add_part(lp,a),add_part(rp,s-a),L+a)
                new[key]=new.get(key,0)+count*comb(s,a)
            for j,kj in enumerate(typ):
                if mask&(1<<j) or s<kj:continue
                t=s-kj
                choices=factorial(s)//factorial(t)
                for a in range(max(0,R+t-m),min(t,m-L)+1):
                    el=list(le);er=list(re);el[j]=a;er[j]=t-a
                    key=(mask|(1<<j),tuple(el),tuple(er),lp,rp,L+a)
                    new[key]=new.get(key,0)+count*choices*comb(t,a)
        states=new;seen+=s
    counts={};full=(1<<r)-1;index=block['index']
    for (mask,le,re,lp,rp,L),v in states.items():
        if mask==full and L==m:
            key=(index[(le,lp)],index[(re,rp)])
            counts[key]=counts.get(key,0)+v
    check(all(counts.get((j,i),0)==v for (i,j),v in counts.items()),
          'Joint probability asymmetry')
    denominator=factorial(N)//factorial(N-k)*comb(N-k,m)
    return counts,denominator


def main():
    default=Path(__file__).resolve().parent/'results'/'sharp_certificate.json.gz'
    path=Path(sys.argv[1]) if len(sys.argv)>1 else default
    out=Path(sys.argv[2]) if len(sys.argv)>2 else Path('independent_output')
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    raw=path.read_bytes();cert=json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)
    check(cert.get('schema')=='cluster-flag-factor-certificate-v1','Wrong schema')
    check(cert.get('order')==15 and tuple(cert.get('target_partition',()))==TARGET,'Wrong problem')
    check(Fraction(cert['bound'])==M,'Not the requested exact bound')
    records=cert['blocks']
    check(len(records)==4 and {tuple(b['type']) for b in records}==set(TYPES),'Wrong types')
    blocks=[prepare_block(next(b for b in records if tuple(b['type'])==t)) for t in TYPES]
    hosts=sorted((tuple(reversed(p)) for p in all_partitions(N)),reverse=True)
    check(len(hosts)==176,'Wrong partition count')
    rows=[]
    all_matrices=[[] for _ in blocks]
    denominators=[None]*len(blocks)
    for li,host in enumerate(hosts):
        total=Fraction(0)
        for bi,block in enumerate(blocks):
            counts,den=joint_counts(host,block)
            all_matrices[bi].append([[i,j,v] for (i,j),v in sorted(counts.items()) if i<=j])
            denominators[bi]=den
            num=sum(count*block['Q'][i][j] for (i,j),count in counts.items())
            total+=Fraction(num,den*block['denominator'])
        residual=M-int(host==TARGET)-total
        check(residual>=0,'Negative residual for '+str(host))
        check(residual==Fraction(cert['residuals'][li]),'Stored residual mismatch')
        rows.append((host,M-int(host==TARGET),total,residual))
    with (out/'independent_residuals.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['partition','bound_minus_indicator','matrix_sum','exact_residual'])
        for host,left,right,res in rows:writer.writerow([' '.join(map(str,host)),str(left),str(right),str(res)])
    independent_model={'schema':'cluster-flag-model-v1','order':15,
        'partitions':[list(h) for h in hosts],
        'blocks':[{'type':list(t),
                   'flags':next(bb['flags'] for bb in records if tuple(bb['type'])==t),
                   'denominator':denominators[bi],
                   'matrices_upper':all_matrices[bi]}
                  for bi,t in enumerate(TYPES)]}
    raw_model=json.dumps(independent_model,separators=(',',':'),sort_keys=True).encode()
    (out/'independent_flag_model.json.gz').write_bytes(gzip.compress(raw_model,mtime=0))
    report={'status':'VERIFIED','model_sha256_uncompressed':hashlib.sha256(raw_model).hexdigest(),'bound':str(M),'identities_checked':len(rows),
            'zero_residuals':sum(r[3]==0 for r in rows),
            'positive_residuals':sum(r[3]>0 for r in rows),
            'negative_residuals':0,'arithmetic':'Exact integers and fractions.Fraction only',
            'counting_algorithm':'Independent component-by-component root/left/right DP',
            'certificate_sha256':hashlib.sha256(raw).hexdigest(),
            'elapsed_seconds':round(time.monotonic()-start,3)}
    (out/'independent_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('INDEPENDENT IDENTITIES: 176 PASS')
    print('RESIDUALS: zero=%d positive=%d negative=0'%(report['zero_residuals'],report['positive_residuals']))
    print('SHARP GLOBAL UPPER BOUND: EXACTLY VERIFIED')
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,KeyError,TypeError,OSError,ArithmeticError) as error:
        print('VERIFICATION FAILED:',error,file=sys.stderr)
        sys.exit(1)
