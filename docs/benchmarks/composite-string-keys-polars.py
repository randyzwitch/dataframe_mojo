import polars as pl,time,os,statistics
n=int(os.environ.get('PROFILE_ROWS','1000000'))
for count in [1,2,3]:
 for shape in [0,1,2]:
  if count==1 and shape==2: continue
  for groups in [10,1000,500000]:
   names=['k'+str(k) for k in range(count)]
   cols={name:pl.Series(name,['value_'+str((i*37%groups)+k) for i in range(n)] if shape==1 or (shape==2 and k%2==1) else [(i*37%groups)+k for i in range(n)],dtype=pl.String if shape==1 or (shape==2 and k%2==1) else pl.Int64) for k,name in enumerate(names)}
   cols['v']=pl.Series('v',[i%101-50 for i in range(n)],dtype=pl.Int64)
   frame=pl.DataFrame(cols); times=[]
   for rep in range(9):
    start=time.perf_counter_ns();result=frame.group_by(names).agg(pl.col('v').sum());elapsed=(time.perf_counter_ns()-start)/1e6
    assert result.height==min(n,groups)
    assert result['v'].sum()==-50 if n==1000000 else True
    if rep:times.append(elapsed)
   print(count,shape,groups,'aggregate',statistics.median(times),'samples',times,flush=True)
