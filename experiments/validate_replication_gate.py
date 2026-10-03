"""Check the second GPU pilot against a shared frozen selection gate."""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.refinement_study import source_hashes,write_json
from experiments.corrected_evaluation import environment
from data.refinement_data import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--gate',required=True);p.add_argument('--pilot',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();gate=json.loads(Path(a.gate).read_text());pilot=json.loads((Path(a.pilot)/'manifest.json').read_text());env=environment()
    if not gate['passed'] or pilot['status']!='completed' or pilot['failures']:raise RuntimeError('Incomplete replicated pilot/gate')
    if gate['source_sha256']!=source_hashes() or pilot['source_sha256']!=source_hashes():raise RuntimeError('Replication source differs')
    if gate['dataset']!=pilot['dataset']:raise RuntimeError('Replication raw data/preprocessing differs')
    if env['dependencies']!=gate['environment']['dependencies'] or env['dependencies']!=pilot['environment']['dependencies']:raise RuntimeError('Replication dependencies differ')
    if any(r['test'] is not None for r in pilot['results']):raise RuntimeError('Replication pilot accessed test')
    if max(r['peak_cuda_bytes'] for r in pilot['results'])>env['gpu_memory_bytes']*.85:raise RuntimeError('Replication pilot lacks memory margin')
    extras={k:importlib.metadata.version(k) for k in ['xgboost','cryptography','matplotlib']}
    if extras!={'xgboost':'2.1.4','cryptography':'44.0.3','matplotlib':'3.9.4'}:raise RuntimeError('Pinned additional dependencies differ')
    write_json(a.output,dict(passed=True,gate_sha256=sha256(a.gate),pilot_sha256=sha256(Path(a.pilot)/'manifest.json'),environment=env,additional_dependencies=extras,
        selection='GTX selected recipes copied unchanged; no hardware-specific re-search',ownership='all ownership hashes independently checked in main runner'))
    print('PASSED hardware replication gate',a.output,flush=True)


if __name__=='__main__':main()
