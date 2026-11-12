from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product
from math import comb, factorial
from fractions import Fraction
import gzip, json, hashlib, argparse, time
from pathlib import Path

ORDER = 15
TARGET = (4, 2) + (1,) * 9
TYPES = ((1,), (3,), (2, 1), (1, 1, 1))
BOUND = Fraction(118216098, 1220703125)

def require(ok, message):
    if not ok:
        raise ValueError(message)

@lru_cache(None)
def partitions(n, cap=None):
    if n == 0:
        return ((),)
    if n < 0:
        return ()
    cap = min(n, n if cap is None else cap)
    return tuple((a,) + tail for a in range(cap, 0, -1)
                 for tail in partitions(n-a, a))

def weak_compositions(n, length):
    if length == 0:
        if n == 0:
            yield ()
    elif length == 1:
        yield (n,)
    else:
        for a in range(n+1):
            for t in weak_compositions(n-a, length-1):
                yield (a,)+t

def flags(n, sigma):
    b = (n-sum(sigma))//2
    require(2*b+sum(sigma)==n, 'Parity error')
    return tuple((a,nu) for total in range(b+1)
                 for a in weak_compositions(total,len(sigma))
                 for nu in partitions(b-total))

def falling(n,k):
    return factorial(n)//factorial(n-k) if n>=k else 0

@lru_cache(None)
def split_cliques(parts):
    states = {((),()):1}
    for s in parts:
        nxt = defaultdict(int)
        for (p,q),w in states.items():
            for a in range(s+1):
                left = tuple(sorted(p+((a,) if a else ()),reverse=True))
                right = tuple(sorted(q+((s-a,) if s-a else ()),reverse=True))
                nxt[left,right] += w*comb(s,a)
        states = nxt
    out = defaultdict(list)
    for (p,q),v in states.items():
        out[sum(p)].append((p,q,v))
    return dict(out)

def place_roots(host,sigma):
    mult = Counter(host)
    sizes = sorted(mult,reverse=True)
    def rec(j,remaining,weight):
        if j==len(sigma):
            untouched=tuple(s for s in sizes for _ in range(mult[s]))
            yield tuple(remaining), untouched, weight
            return
        for s in sizes:
            if mult[s] and s>=sigma[j]:
                w=mult[s]*falling(s,sigma[j])
                mult[s]-=1
                yield from rec(j+1,remaining+[s-sigma[j]],weight*w)
                mult[s]+=1
    yield from rec(0,[],1)

def count_matrix(host,sigma,basis):
    index={f:i for i,f in enumerate(basis)}
    b=(sum(host)-sum(sigma))//2
    out=defaultdict(int)
    root_total=0
    for remaining,tail,w0 in place_roots(host,sigma):
        root_total+=w0
        tail_splits=split_cliques(tail)
        for left in product(*(range(s+1) for s in remaining)):
            need=b-sum(left)
            if need not in tail_splits: continue
            right=tuple(s-a for s,a in zip(remaining,left))
            w=w0
            for s,a in zip(remaining,left): w*=comb(s,a)
            for lp,rp,v in tail_splits[need]:
                out[index[left,lp],index[right,rp]]+=w*v
    require(all(out.get((j,i),0)==w for (i,j),w in out.items()),'Asymmetric counts')
    require(sum(out.values())==root_total*comb(2*b,b),'Counting normalization')
    return [[i,j,w] for (i,j),w in sorted(out.items()) if i<=j]

def dump_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(obj,separators=(',',':'),sort_keys=True).encode()
    path.write_bytes(gzip.compress(raw,mtime=0) if path.suffix=='.gz' else raw)

def load_json(path):
    raw=Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)

def build_model(n=ORDER):
    hosts=partitions(n)
    blocks=[]
    for sigma in TYPES:
        fs=flags(n,sigma)
        d=falling(n,sum(sigma))*comb(n-sum(sigma),(n-sum(sigma))//2)
        matrices=[count_matrix(h,sigma,fs) for h in hosts]
        blocks.append(dict(type=list(sigma),flags=[[list(a),list(nu)] for a,nu in fs],
                           denominator=d,matrices_upper=matrices))
        print(f'MODEL type={sigma} flags={len(fs)} matrices={len(hosts)} denominator={d}',flush=True)
    return dict(schema='cluster-flag-model-v1',order=n,partitions=[list(h) for h in hosts],blocks=blocks)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results/model.json.gz')
    args=ap.parse_args();start=time.monotonic()
    model=build_model();dump_json(args.out,model)
    print('MODEL_SECONDS',time.monotonic()-start,flush=True)
