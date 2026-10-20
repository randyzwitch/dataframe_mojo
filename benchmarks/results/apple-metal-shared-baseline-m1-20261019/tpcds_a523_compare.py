#!/usr/bin/env python3
"""Read-only exact/keyed/multiset comparison of completed TPCDS base exports."""
import hashlib,json,re,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import polars as pl

ROOT=Path(sys.argv[1]); LOGROOT=ROOT.parent/"logs"; OUT=Path(sys.argv[2]); SCRIPT_SHA=sys.argv[3]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def schema(p):
 d=json.loads(p.read_text())
 return {"rows":d["rows"],"columns":[{"name":x["name"],"dtype":x["dtype"]["name"],"encoded":x["dtype"].get("encoded")} for x in d["columns"]]}
def exports(log):
 out={}
 for line in log.read_text().splitlines():
  if line.startswith('answer_export\t'):
   x=json.loads(line.split('\t',1)[1]); out[x['engine']]=(Path(x['path']),Path(x['schema_path']))
 return out
def is_float_dtype(dtype): return dtype in (pl.Float32,pl.Float64)
def float_parts(value,dtype):
 if value is None: return None,None,None
 typ=np.float32 if dtype==pl.Float32 else np.float64
 bits_type=np.uint32 if dtype==pl.Float32 else np.uint64
 width=32 if dtype==pl.Float32 else 64
 raw=int(np.asarray([value],dtype=typ).view(bits_type)[0])
 sign=1<<(width-1); mask=(1<<width)-1
 ordered=(~raw)&mask if raw&sign else raw|sign
 return raw,ordered,float(value)
def row_token(df,row):
 token=[]
 for name,dtype in zip(df.columns,df.dtypes):
  value=df[name][row]
  if value is None: token.append((name,'null')); continue
  if is_float_dtype(dtype):
   raw,_,_=float_parts(value,dtype); token.append((name,'floatbits',str(dtype),raw))
  else:
   token.append((name,str(dtype),value))
 return tuple(token)
def nonfloat_signature(df,row):
 out=[]
 for name,dtype in zip(df.columns,df.dtypes):
  if is_float_dtype(dtype): continue
  v=df[name][row]
  try: hash(v)
  except TypeError: v=repr(v)
  out.append((name,str(dtype),v))
 return tuple(out)
def metric_pairs(left,right,li,ri,name,dtype):
 pairs=[(left[name][i],right[name][j]) for i,j in zip(li,ri)]
 if is_float_dtype(dtype):
  raw_diffs=val_diffs=nan_pairs=0; ulps=[]; absdiff=[]; reldiff=[]
  for a,b in pairs:
   if a is None or b is None:
    if a!=b: raw_diffs+=1; val_diffs+=1
    continue
   ra,oa,fa=float_parts(a,dtype); rb,ob,fb=float_parts(b,dtype)
   if ra!=rb: raw_diffs+=1
   if np.isnan(fa) or np.isnan(fb):
    if np.isnan(fa) and np.isnan(fb): nan_pairs+=1
    elif fa!=fb: val_diffs+=1
    continue
   if fa!=fb: val_diffs+=1
   if np.isfinite(fa) and np.isfinite(fb):
    ulps.append(abs(oa-ob)); ad=abs(fa-fb); absdiff.append(ad)
    reldiff.append(ad/max(abs(fa),np.finfo(np.float64).tiny))
  return {"physical_dtype":str(dtype),"bit_differences":raw_diffs,"numeric_differences":val_diffs,"compared_rows":len(pairs),"nan_pairs":nan_pairs,"nulls":{"left":left[name].null_count(),"right":right[name].null_count()},"ulp_max_finite":max(ulps,default=0),"ulp_over_1_finite":sum(v>1 for v in ulps),"absolute_error_max":max(absdiff,default=0.0),"relative_error_max":max(reldiff,default=0.0),"exact":raw_diffs==0}
 if isinstance(dtype,pl.Decimal):
  diffs=[]; units=[]; absd=[]
  for a,b in pairs:
   diffs.append(a!=b)
   if a is not None and b is not None:
    scale=dtype.scale; units.append(int((a-b).scaleb(scale))); absd.append(abs(a-b))
  return {"physical_dtype":str(dtype),"exact_decimal_differences":sum(diffs),"compared_rows":len(pairs),"decimal_scale":dtype.scale,"unscaled_integer_unit_differences":sum(v!=0 for v in units),"unscaled_abs_max":max((abs(v) for v in units),default=0),"absolute_error_max":str(max(absd,default=0)),"exact":not any(diffs)}
 diffs=sum(a!=b for a,b in pairs)
 return {"physical_dtype":str(dtype),"value_differences":diffs,"compared_rows":len(pairs),"nulls":{"left":left[name].null_count(),"right":right[name].null_count()},"exact":diffs==0}
def compare(a,b):
 schemas_equal=schemas(a)==schemas(b)
 n=min(a.height,b.height)
 exact_row_order=(a.height==b.height and all(row_token(a,i)==row_token(b,i) for i in range(n)))
 ac=Counter(row_token(a,i) for i in range(a.height)); bc=Counter(row_token(b,i) for i in range(b.height))
 exact_row_multiset=ac==bc
 nonfloat_a=[nonfloat_signature(a,i) for i in range(a.height)]
 nonfloat_b=[nonfloat_signature(b,i) for i in range(b.height)]
 ca,cb=Counter(nonfloat_a),Counter(nonfloat_b)
 nonfloat_multiset_equal=ca==cb
 nonfloat_columns=[c for c,d in zip(a.columns,a.dtypes) if not is_float_dtype(d)]
 unique_nonfloat=(bool(nonfloat_columns) and len(ca)==a.height and len(cb)==b.height and ca==cb)
 alignment='unique_nonfloat_signature' if unique_nonfloat else 'nonfloat_signature_multiset_per_float_column' if nonfloat_multiset_equal else 'row_position_only_nonfloat_signatures_differ'
 paired=[]
 duplicate_group_count=0
 if unique_nonfloat:
  posb={key:i for i,key in enumerate(nonfloat_b)}
  posa={key:i for i,key in enumerate(nonfloat_a)}
  # align in left row order
  paired=[(i,posb[key]) for i,key in enumerate(nonfloat_a)]
 elif nonfloat_multiset_equal and nonfloat_columns:
  # Preserve multiplicity; within each duplicate non-float signature, sort float values per column before pairing.
  ga=defaultdict(list); gb=defaultdict(list)
  for i,k in enumerate(nonfloat_a): ga[k].append(i)
  for i,k in enumerate(nonfloat_b): gb[k].append(i)
  duplicate_group_count=sum(len(v)>1 for v in ga.values())
  for k in ga:
   la,lb=ga[k],gb[k]
   for col,dtype in zip(a.columns,a.dtypes):
    if is_float_dtype(dtype):
     def sortkey(i):
      v=a[col][i]
      if v is None: return (0,0)
      _,o,_=float_parts(v,dtype); return (1,o)
     la=sorted(la,key=sortkey)
     def sortkeyb(i):
      v=b[col][i]
      if v is None: return (0,0)
      _,o,_=float_parts(v,dtype); return (1,o)
     lb=sorted(lb,key=sortkeyb)
     # Column-wise sorted pairing records scalar value bags, not cross-column row association.
     for i,j in zip(la,lb): paired.append((i,j))
   # For rows with no float columns, or if no floats, add one pairing per group.
   if not any(is_float_dtype(d) for d in a.dtypes): paired.extend(zip(la,lb))
  # pairs now contain one pairing per float column; de-duplicate for nonfloat metrics.
  paired=list(dict.fromkeys(paired))
 else:
  paired=[(i,i) for i in range(n)]
 nonfloat_column_stats={}
 for col,dtype in zip(a.columns,a.dtypes):
  if is_float_dtype(dtype): continue
  lvals=Counter(a[col].to_list()); rvals=Counter(b[col].to_list())
  nonfloat_column_stats[col]={"physical_dtype":str(dtype),"column_multiset_equal":lvals==rvals,"left_rows":a.height,"right_rows":b.height,"symmetric_count_delta":sum((lvals-rvals).values())+sum((rvals-lvals).values())}
 float_stats={}
 for col,dtype in zip(a.columns,a.dtypes):
  if not is_float_dtype(dtype): continue
  if nonfloat_multiset_equal and nonfloat_columns and not unique_nonfloat:
   # Independently sort the float column within each non-float signature and pair duplicates by value order.
   pairs=[]
   ga=defaultdict(list);gb=defaultdict(list)
   for i,k in enumerate(nonfloat_a):ga[k].append(i)
   for i,k in enumerate(nonfloat_b):gb[k].append(i)
   for k,la0 in ga.items():
    lb0=gb[k]
    la=sorted(la0,key=lambda i:(a[col][i] is not None, float_parts(a[col][i],dtype)[1] if a[col][i] is not None else 0))
    lb=sorted(lb0,key=lambda i:(b[col][i] is not None, float_parts(b[col][i],dtype)[1] if b[col][i] is not None else 0))
    pairs.extend((i,j) for i,j in zip(la,lb))
  elif not unique_nonfloat:
   # No trustworthy non-float key: preserve duplicates by comparing global float-value multisets.
   la=sorted(range(a.height),key=lambda i:(a[col][i] is not None, float_parts(a[col][i],dtype)[1] if a[col][i] is not None else 0))
   lb=sorted(range(b.height),key=lambda i:(b[col][i] is not None, float_parts(b[col][i],dtype)[1] if b[col][i] is not None else 0))
   pairs=list(zip(la,lb))
  else:
   pairs=paired
  # Reuse exact scalar metric; if duplicate signatures require per-column matching, pass an adapter pair list directly.
  lidx=[i for i,j in pairs]; ridx=[j for i,j in pairs]
  # metric_pairs expects parallel position lists.
  float_stats[col]=metric_pairs(a,b,lidx,ridx,col,dtype)
 # Compare non-float signature row order exactly, if available.
 return {"schema_equal":schemas_equal,"left_schema":schemas(a),"right_schema":schemas(b),"row_count_equal":a.height==b.height,"rows":{"left":a.height,"right":b.height},"exact_ordered_rows_equal":exact_row_order,"exact_row_multiset_equal":exact_row_multiset,"nonfloat_columns":nonfloat_columns,"nonfloat_column_multiset":nonfloat_column_stats,"nonfloat_signature_multiset_equal":nonfloat_multiset_equal,"nonfloat_signature_order_equal":nonfloat_a==nonfloat_b,"nonfloat_signature_unique_both":unique_nonfloat,"value_alignment":alignment,"duplicate_nonfloat_signature_groups":duplicate_group_count,"float_columns":float_stats}

def schemas(df):
 return [(n,str(d)) for n,d in zip(df.columns,df.dtypes)]
def main():
 report={"answer_root":str(ROOT),"log_root":str(LOGROOT),"coverage_run":"metal-868-dev-coverage-a52334e7","analysis":{"script_path":"/tmp/tpcds_a523_compare.py","script_sha256":SCRIPT_SHA,"python_environment":"pixi --locked, -e oracle, manifest ~/GitHub/dataframe_mojo-metal-868/pixi.toml","scope":"completed tpcds-base logs classified wrong_answer only; no GPU/compiler/input","duplicate_handling":"exact full-row Counters preserve multiplicity; if no unique nonfloat signature, float columns are matched as per-column multisets within nonfloat signature groups"},"cases":{}}
 logs=sorted(LOGROOT.glob('tpcds-base-q*.txt'),key=lambda p:int(re.search(r'q(\d+)',p.name).group(1)))
 for log in logs:
  text=log.read_text()
  if not any(line.startswith('coverage\t') and '\twrong_answer\t' in line for line in text.splitlines()): continue
  case=log.stem; xp=exports(log); entry={"log_path":str(log),"log_sha256":sha(log),"coverage_classification":"wrong_answer","exports":{},"comparisons":{}}
  for engine,(p,sp) in xp.items():
   entry['exports'][engine]={"parquet_path":str(p),"parquet_sha256":sha(p),"parquet_bytes":p.stat().st_size,"schema_path":str(sp),"schema_sha256":sha(sp),"schema_bytes":sp.stat().st_size,"schema":schema(sp)}
  if 'cpu' not in xp:
   entry['status']='CPU export missing';report['cases'][case]=entry;continue
  cpu=pl.read_parquet(xp['cpu'][0])
  for engine in ('metal','cpu_repeat'):
   if engine in xp:
    other=pl.read_parquet(xp[engine][0]); result=compare(cpu,other); result['encoded_schema_equal']=schema(xp['cpu'][1])==schema(xp[engine][1]); entry['comparisons'][f"cpu0_vs_{engine}0"]=result
  entry['status']='compared exact available exports; exact mismatches remain failures'
  report['cases'][case]=entry
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
 print(OUT)
def exports(log):
 out={}
 for line in log.read_text().splitlines():
  if line.startswith('answer_export\t'):
   x=json.loads(line.split('\t',1)[1]); out[x['engine']]=(Path(x['path']),Path(x['schema_path']))
 return out
main()
