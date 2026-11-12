#!/usr/bin/env python3
"""Rebuild, search, rationalize and independently verify from an empty directory."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',default='results',help='New, empty output directory')
    args=ap.parse_args();out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty; use a new --out path')
    out.mkdir(parents=True,exist_ok=True);logs=out/'logs';logs.mkdir()
    env=os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env[name]='1'
    env['PYTHONHASHSEED']='0'
    cert=out/'sharp_certificate.json.gz'
    steps=[
        ('model',['flag_model.py','--out',str(out/'model.json.gz')]),
        ('search_sharp',['search_sdp.py','--model',str(out/'model.json.gz'),'--out',str(out),'--mode','sharp']),
        ('recover',['recover_certificate.py','--out',str(out)]),
        ('verify',['verify_certificate.py','--certificate',str(cert),'--out',str(out/'verified')]),
        ('verify_independent',['verify_independent.py',str(cert),str(out/'independent')]),
        ('local',['check_local.py','--out',str(out/'local')]),
        ('tests',['regression_tests.py','--certificate',str(cert),'--out',str(out/'tests')]),
        ('verify_python_O',['-O','verify_certificate.py','--certificate',str(cert),'--out',str(out/'verified_python_O')])]
    reports=[];start=time.monotonic()
    sources={p.name:digest(p) for p in sorted(HERE.glob('*.py'))}
    for name,arguments in steps:
        cmd=[sys.executable]+arguments
        script_at=2 if arguments[0]=='-O' else 1
        cmd[script_at]=str(HERE/cmd[script_at])
        print('RUN',name,flush=True);begin=time.monotonic()
        with (logs/(name+'.log')).open('w',encoding='utf-8') as stream:
            process=subprocess.run(cmd,cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record=dict(step=name,command=cmd,exit_status=process.returncode,elapsed_seconds=time.monotonic()-begin,log=name+'.log')
        reports.append(record)
        if process.returncode:
            print((logs/(name+'.log')).read_text()[-8000:],file=sys.stderr)
            raise RuntimeError('Step failed: '+name)
        print('PASS',name,flush=True)
    from flag_model import load_json
    original=load_json(out/'model.json.gz')
    primary=load_json(out/'verified'/'recomputed_model.json.gz')
    independent=load_json(out/'independent'/'independent_flag_model.json.gz')
    if original!=primary or original!=independent:
        raise ValueError('Exact count matrices disagree')
    verify=load_json(out/'verified'/'report.json')
    report=dict(status='COMPLETE_EXACT_VERIFIED',old_certificate_read=False,
        process='empty-directory fresh search; exact recovery; two exact verifiers',
        steps=reports,source_sha256=sources,all_704_count_matrices_identical=True,
        bound=verify['bound'],zero_residuals=verify['zero_residuals'],positive_residuals=verify['positive_residuals'],
        certificate_sha256=digest(cert),elapsed_seconds=time.monotonic()-start,
        python=sys.version,platform=platform.platform())
    (out/'run_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('COMPLETE_EXACT_VERIFIED',report['bound'],'704 matrices identical;',report['zero_residuals'],'zero;',report['positive_residuals'],'positive',flush=True)

if __name__=='__main__':main()
