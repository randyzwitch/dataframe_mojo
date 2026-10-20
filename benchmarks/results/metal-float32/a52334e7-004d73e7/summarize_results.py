"""Report complete paired cases without turning incomplete runs into speedup claims."""
import collections
import hashlib
import itertools
import json
import statistics
import sys
from pathlib import Path


def summarize(data):
    selection = data['selection']
    rounds, reps = selection['rounds'], selection['reps']
    cases = list(itertools.product(selection['rows'], selection['workloads'], selection['variants']))
    expected = set(itertools.product(range(rounds), ('baseline', 'mojo'), cases))
    observed = collections.Counter((r['round'], r['runtime'], (r['rows'], r['workload'], r['variant'])) for r in data['runs'])
    faults = []
    if set(observed) != expected or any(n != 1 for n in observed.values()):
        faults.append('Missing, duplicate or unexpected worker records')
    for r in data['runs']:
        if r['status'] != 'ok' or not r.get('validated') or len(r.get('samples', [])) != reps:
            faults.append(f"Invalid worker: {r['runtime']} round={r['round']} rows={r['rows']} {r['workload']} {r['variant']} status={r['status']}")
        elif any(s['metal_ns'] <= 0 or s['cpu_ns'] <= 0 for s in r['samples']):
            faults.append('Nonpositive timing sample')
    if faults:
        return {'complete': False, 'faults': faults, 'workers_observed': len(data['runs']), 'workers_expected': len(expected)}
    groups = {(runtime, case): [] for runtime in ('baseline', 'mojo') for case in cases}
    for r in data['runs']:
        groups[(r['runtime'], (r['rows'], r['workload'], r['variant']))].append(r)
    records = []
    for case in cases:
        base = sorted(groups[('baseline', case)], key=lambda r:r['round'])
        candidate = sorted(groups[('mojo', case)], key=lambda r:r['round'])
        med = lambda runs, field: statistics.median(s[field] for r in runs for s in r['samples'])
        b, c = med(base, 'metal_ns'), med(candidate, 'metal_ns')
        round_ratios = [med([br], 'metal_ns') / med([cr], 'metal_ns') for br, cr in zip(base,candidate)]
        profiles = {}
        for name, runs in [('baseline',base), ('candidate',candidate)]:
            profiles[name] = {}
            for field in ('kernel_launches','peak_requested_bytes','upload_bytes','download_bytes','host_result_bytes','synchronizations'):
                values = [r.get('profile',{}).get(field) for r in runs]
                profiles[name][field] = statistics.median(values) if all(isinstance(v,(int,float)) for v in values) else None
            profiles[name]['routes'] = dict(collections.Counter(r.get('profile',{}).get('region_0_grouped_int_algorithm', r.get('profile',{}).get('region_0_algorithm','missing')) for r in runs))
        records.append(dict(rows=case[0],workload=case[1],variant=case[2],baseline_ms=b/1e6,candidate_ms=c/1e6,candidate_cpu_ms=med(candidate,'cpu_ns')/1e6,speedup=b/c,round_speedups=round_ratios,baseline_first_use_ms=statistics.median(r['first_use_ns']['metal'] for r in base)/1e6,candidate_first_use_ms=statistics.median(r['first_use_ns']['metal'] for r in candidate)/1e6,profiles=profiles))
    return {'complete':True,'workers':len(data['runs']),'cells':records}


def render(result):
    if not result['complete']:
        return 'Incomplete paired run; no speedup conclusion.\n\n'+'\n'.join('- '+f for f in result['faults'])+'\n'
    lines=['Paired complete-query warm medians. A ratio above 1 means the candidate was faster than the baseline. Per-round ranges show variability; they are not confidence intervals. CPU timings and profile counters are reported separately in the JSON.','', '| Rows | Workload | Variant | Baseline ms | Candidate ms | CPU ms | Ratio | Round ratio range |','|---:|---|---|---:|---:|---:|---:|---:|']
    for r in result['cells']:
        lines.append(f"| {r['rows']} | {r['workload']} | {r['variant']} | {r['baseline_ms']:.3f} | {r['candidate_ms']:.3f} | {r['candidate_cpu_ms']:.3f} | {r['speedup']:.3f} | {min(r['round_speedups']):.3f}–{max(r['round_speedups']):.3f} |")
    lines += ['', '| Rows | Workload | Variant | First use baseline/candidate ms | Requested peak baseline/candidate bytes | Launches baseline/candidate |', '|---:|---|---|---:|---:|---:|']
    for r in result['cells']:
        b, c = r['profiles']['baseline'], r['profiles']['candidate']
        lines.append(f"| {r['rows']} | {r['workload']} | {r['variant']} | {r['baseline_first_use_ms']:.3f}/{r['candidate_first_use_ms']:.3f} | {b['peak_requested_bytes']}/{c['peak_requested_bytes']} | {b['kernel_launches']}/{c['kernel_launches']} |")
    lines += ['', 'First use is a fresh query process with runtime initialization; it is not a cold-machine measurement.', '', 'These are measured development cases, not a general GPU-over-CPU claim. Collection timings include staging and result construction. Profile memory is requested allocation accounting, not measured device residency. Retain build receipts and raw samples with this derived report.','']
    return '\n'.join(lines)

if __name__ == '__main__':
    source, destination = map(Path,sys.argv[1:])
    raw=source.read_bytes()
    data=json.loads(raw)
    result=summarize(data)
    result['source_results_sha256']=hashlib.sha256(raw).hexdigest()
    result['selection']=data.get('selection',{})
    result['binary_receipts']=data.get('metadata',{}).get('binary_receipts',{})
    result['environment']={k:data.get('metadata',{}).get(k) for k in ('platform','cpu','physical_cores','memory_bytes','mojo_version','dataframe_threads','compile_flags')}
    destination.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    destination.with_suffix('.md').write_text(render(result))
