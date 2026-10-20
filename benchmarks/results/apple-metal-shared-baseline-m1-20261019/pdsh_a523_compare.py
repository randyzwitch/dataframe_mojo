#!/usr/bin/env python3
"""Read-only exact comparison of completed PDSH CPU/Metal exports."""
import hashlib, json, math, re, sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
import numpy as np
import polars as pl

ROOT=Path(sys.argv[1]); LOGROOT=ROOT.parent/"logs"; OUT=Path(sys.argv[2]); SCRIPT_SHA=sys.argv[3]
CASES=["pdsh-base-q1","pdsh-base-q5","pdsh-base-q6","pdsh-base-q7","pdsh-base-q8","pdsh-base-q9","pdsh-base-q10","pdsh-base-q14","pdsh-base-q15","pdsh-base-q17","pdsh-decimal-q8"]
CANDIDATE_KEYS={
 "pdsh-base-q1":["l_returnflag","l_linestatus"],
 "pdsh-base-q5":["n_name"],
 "pdsh-base-q6":[], "pdsh-base-q7":["supp_nation","cust_nation","l_year"],
 "pdsh-base-q8":["o_year"], "pdsh-base-q9":["nation","o_year"],
 "pdsh-base-q10":["c_custkey"], "pdsh-base-q14":[], "pdsh-base-q15":[],
 "pdsh-base-q17":[], "pdsh-decimal-q8":["o_year"],
}
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()
def schema(path):
 d=json.loads(path.read_text())
 return {"rows":d["rows"],"columns":[{"name":x["name"],"dtype":x["dtype"]["name"],"encoded":x["dtype"].get("encoded")} for x in d["columns"]]}
def export_paths(log):
 result={}
 for line in log.read_text().splitlines():
  if not line.startswith("answer_export\t"): continue
  rec=json.loads(line.split("\t",1)[1]); result[rec["engine"]]=(Path(rec["path"]),Path(rec["schema_path"]))
 return result
def keys(df,cands):
 cols=[k for k in cands if k in df.columns]
 if not cols: return [],[],False
 vals=[tuple(r) for r in df.select(cols).rows()]
 return cols,vals,len(set(vals))==len(vals)
def ordered_bits(bits):
 sign=1<<63
 return (~bits)&((1<<64)-1) if bits&sign else bits|sign
def compare(a,b,requested_keys):
 kc,ka,ku=keys(a,requested_keys); _,kb,ku_b=keys(b,requested_keys)
 aligned=bool(kc) and ku and ku_b and set(ka)==set(kb)
 if aligned:
  ai={k:i for i,k in enumerate(ka)}; bi={k:i for i,k in enumerate(kb)}; row_pairs=[(ai[k],bi[k]) for k in ka]
 else:
  ai={};bi={};row_pairs=[(i,i) for i in range(min(a.height,b.height))]
 out={"rows":{"left":a.height,"right":b.height},"candidate_key_columns":kc,
      "key_columns_unique_both":bool(kc) and ku and ku_b,
      "key_sets_equal":set(ka)==set(kb) if kc and ku and ku_b else None,
      "key_order_equal":ka==kb if kc and ku and ku_b else None,
      "key_positions_differ":sum(x!=y for x,y in zip(ka,kb)) if kc and ku and ku_b else None,
      "value_alignment":"unique-key-aligned" if aligned else "row-position-only (no validated unique key)",
      "columns":{}}
 for name,dtype in zip(a.columns,a.dtypes):
  av=a[name].to_list(); bv=b[name].to_list()
  pairs=[(av[i],bv[j]) for i,j in row_pairs]
  if dtype in (pl.Float32,pl.Float64):
   x=np.asarray([u for u,v in pairs],dtype=np.float64); y=np.asarray([v for u,v in pairs],dtype=np.float64)
   ux,uy=x.view(np.uint64),y.view(np.uint64); bitdiff=ux!=uy; finite=np.isfinite(x)&np.isfinite(y)
   ulps=[abs(ordered_bits(int(m))-ordered_bits(int(n))) for m,n in zip(ux[finite],uy[finite])]
   ad=np.abs(x-y); rd=ad/np.maximum(np.abs(x),np.finfo(np.float64).tiny)
   out["columns"][name]={"physical_dtype":str(dtype),"exact_bit_differences":int(bitdiff.sum()),"compared_rows":len(x),"numeric_differences":int((x!=y).sum()),"nulls":{"left":a[name].null_count(),"right":b[name].null_count()},"ulp_max_finite":max(ulps,default=0),"ulp_over_1_finite":sum(v>1 for v in ulps),"absolute_error_max":float(np.nanmax(ad)) if len(ad) else 0.0,"relative_error_max":float(np.nanmax(rd)) if len(rd) else 0.0,"exact":not bool(bitdiff.any())}
  elif isinstance(dtype,pl.Decimal):
   diffs=[]; absd=[]
   for u,v in pairs:
    if u is None or v is None:
     diffs.append(u!=v)
    else:
     diffs.append(u!=v); absd.append(abs(u-v))
   scale=dtype.scale
   units=[int((u-v).scaleb(scale)) for u,v in pairs if u is not None and v is not None]
   out["columns"][name]={"physical_dtype":str(dtype),"exact_decimal_differences":sum(diffs),"compared_rows":len(pairs),"decimal_scale":scale,"unscaled_integer_unit_differences":sum(v!=0 for v in units),"unscaled_abs_max":max((abs(v) for v in units),default=0),"absolute_error_max":str(max(absd,default=Decimal(0))),"nulls":{"left":a[name].null_count(),"right":b[name].null_count()},"exact":not any(diffs)}
  else:
   diffs=sum(u!=v for u,v in pairs)
   out["columns"][name]={"physical_dtype":str(dtype),"value_differences":int(diffs),"compared_rows":len(pairs),"nulls":{"left":a[name].null_count(),"right":b[name].null_count()},"exact":not bool(diffs)}
 return out
report={"answer_root":str(ROOT),"log_root":str(LOGROOT),"coverage_run":"metal-868-dev-coverage-a52334e7","analysis":{"script_path":"/tmp/pdsh_a523_compare.py","script_sha256":SCRIPT_SHA,"python_environment":"pixi --locked, -e oracle, manifest ~/GitHub/dataframe_mojo-metal-868/pixi.toml","scope":"completed log-classified wrong_answer exports only; no GPU/compiler/input"},"cases":{}}
for case in CASES:
 log=LOGROOT/f"{case}.txt"
 if not log.is_file(): report["cases"][case]={"status":"log missing"}; continue
 paths=export_paths(log)
 logtext=log.read_text()
 wrong=any(line.startswith("coverage\t") and "\twrong_answer\t" in line for line in logtext.splitlines())
 item={"log_path":str(log),"log_sha256":sha(log),"coverage_classification":"wrong_answer" if wrong else "not_wrong_answer","exports":{},"comparisons":{}}
 for engine,(p,sp) in paths.items():
  # Accept existing outputs from this exact logged run; hashes guard all inputs.
  item["exports"][engine]={"parquet_path":str(p),"parquet_sha256":sha(p),"parquet_bytes":p.stat().st_size,"schema_path":str(sp),"schema_sha256":sha(sp),"schema_bytes":sp.stat().st_size,"schema":schema(sp)}
 if "cpu" not in paths:
  item["status"]="CPU export missing"; report["cases"][case]=item; continue
 cpu=pl.read_parquet(paths["cpu"][0]); candidate=CANDIDATE_KEYS[case]
 for engine in ("metal","cpu_repeat"):
  if engine in paths:
   other=pl.read_parquet(paths[engine][0])
   item["comparisons"][f"cpu0_vs_{engine}0"]={"schema_equal":schema(paths["cpu"][1])==schema(paths[engine][1]),**compare(cpu,other,candidate)}
 item["status"]="compared exact available exports; differences remain failures"
 report["cases"][case]=item
# Compare the same q8 projection across the base and decimal-input variants.
report["base_vs_decimal_q8"] = {}
for engine in ("cpu", "cpu_repeat", "metal"):
 base_export=report["cases"]["pdsh-base-q8"]["exports"].get(engine)
 dec_export=report["cases"]["pdsh-decimal-q8"]["exports"].get(engine)
 if base_export and dec_export:
  base_df=pl.read_parquet(base_export["parquet_path"])
  dec_df=pl.read_parquet(dec_export["parquet_path"])
  report["base_vs_decimal_q8"][engine]={"left_path":base_export["parquet_path"],"left_sha256":base_export["parquet_sha256"],"right_path":dec_export["parquet_path"],"right_sha256":dec_export["parquet_sha256"],"schema_equal":base_export["schema"]==dec_export["schema"],**compare(base_df,dec_df,["o_year"])}
OUT.write_text(json.dumps(report,indent=2,sort_keys=True,default=lambda x:x.item() if hasattr(x,"item") else str(x))+"\n")
print(OUT)
