#ifndef DFM_MOMENTS_SOURCE_H
#define DFM_MOMENTS_SOURCE_H

// Integer-only binary64 versions of VarState and CoMomentState. Every
// operation and association follows dataframe/reductions.mojo.
static const char *dfm_moments_source = R"metal(
struct VarianceState { ulong count, mean, m2; };
// Terriberry updates and Pebay merges for third and fourth central moments.
struct CentralMomentState { ulong count, mean, m2, m3, m4; };
inline CentralMomentState central_add(CentralMomentState s, ulong value) {
  ulong before=f64_from_unsigned(s.count),n=f64_from_unsigned(++s.count);
  ulong two=f64_from_unsigned(2),three=f64_from_unsigned(3);
  ulong four=f64_from_unsigned(4),six=f64_from_unsigned(6);
  ulong delta=f64_sub(value,s.mean),dn=f64_div(delta,n),dn2=f64_mul(dn,dn);
  ulong term=f64_mul(f64_mul(delta,dn),before);
  s.mean=f64_add(s.mean,dn);
  ulong polynomial=f64_add(f64_sub(f64_mul(n,n),f64_mul(three,n)),three);
  s.m4=f64_add(s.m4,f64_sub(f64_add(f64_mul(f64_mul(term,dn2),polynomial),
      f64_mul(f64_mul(six,dn2),s.m2)),f64_mul(f64_mul(four,dn),s.m3)));
  s.m3=f64_add(s.m3,f64_sub(f64_mul(f64_mul(term,dn),f64_sub(n,two)),
      f64_mul(f64_mul(three,dn),s.m2)));
  s.m2=f64_add(s.m2,term);
  return s;
}
inline CentralMomentState central_merge(CentralMomentState s, CentralMomentState o) {
  if(!o.count)return s;
  if(!s.count)return o;
  ulong a=f64_from_unsigned(s.count),b=f64_from_unsigned(o.count),n=f64_add(a,b);
  ulong d=f64_sub(o.mean,s.mean),d2=f64_mul(d,d),n2=f64_mul(n,n);
  ulong three=f64_from_unsigned(3),four=f64_from_unsigned(4),six=f64_from_unsigned(6);
  ulong m2=f64_add(f64_add(s.m2,o.m2),f64_div(f64_mul(f64_mul(d2,a),b),n));
  ulong m3=f64_add(f64_add(f64_add(s.m3,o.m3),
      f64_div(f64_mul(f64_mul(f64_mul(f64_mul(d2,d),a),b),f64_sub(a,b)),n2)),
      f64_div(f64_mul(f64_mul(three,d),f64_sub(f64_mul(a,o.m2),f64_mul(b,s.m2))),n));
  ulong polynomial=f64_add(f64_sub(f64_mul(a,a),f64_mul(a,b)),f64_mul(b,b));
  ulong m4=f64_add(f64_add(f64_add(f64_add(s.m4,o.m4),
      f64_div(f64_mul(f64_mul(f64_mul(f64_mul(d2,d2),a),b),polynomial),f64_mul(n2,n))),
      f64_div(f64_mul(f64_mul(six,d2),f64_add(f64_mul(f64_mul(a,a),o.m2),
          f64_mul(f64_mul(b,b),s.m2))),n2)),
      f64_div(f64_mul(f64_mul(four,d),f64_sub(f64_mul(a,o.m3),f64_mul(b,s.m3))),n));
  s.mean=f64_add(s.mean,f64_div(f64_mul(d,b),n));
  s.m2=m2;s.m3=m3;s.m4=m4;s.count+=o.count;
  return s;
}
inline bool central_valid(ulong count, bool kurtosis, ulong flags) {
  return count && ((flags&1UL) || count>=(kurtosis?4UL:3UL));
}
inline ulong central_finish(CentralMomentState s, bool kurtosis, ulong flags) {
  ulong n=f64_from_unsigned(s.count),one=f64_from_unsigned(1);
  ulong two=f64_from_unsigned(2),three=f64_from_unsigned(3),six=f64_from_unsigned(6);
  if(!kurtosis){
    ulong g=f64_div(f64_mul(f64_sqrt(n),s.m3),f64_mul(s.m2,f64_sqrt(s.m2)));
    return flags&1UL?g:f64_div(f64_mul(g,f64_sqrt(f64_mul(n,f64_sub(n,one)))),f64_sub(n,two));
  }
  ulong excess=f64_sub(f64_div(f64_mul(n,s.m4),f64_mul(s.m2,s.m2)),three);
  if(!(flags&1UL))excess=f64_div(f64_mul(f64_add(f64_mul(f64_add(n,one),excess),six),
      f64_sub(n,one)),f64_mul(f64_sub(n,two),f64_sub(n,three)));
  return flags&2UL?excess:f64_add(excess,three);
}
inline VarianceState variance_add(VarianceState s, ulong value) {
  ++s.count;
  ulong delta=f64_sub(value,s.mean);
  s.mean=f64_add(s.mean,f64_div(delta,f64_from_unsigned(s.count)));
  s.m2=f64_add(s.m2,f64_mul(delta,f64_sub(value,s.mean)));
  return s;
}
inline VarianceState variance_merge(VarianceState s, VarianceState other) {
  if(!other.count)return s;
  if(!s.count)return other;
  ulong total=s.count+other.count;
  ulong delta=f64_sub(other.mean,s.mean);
  ulong weight=f64_div(f64_from_unsigned(other.count),f64_from_unsigned(total));
  s.mean=f64_add(s.mean,f64_mul(delta,weight));
  ulong correction=f64_div(f64_mul(f64_mul(f64_mul(delta,delta),
      f64_from_unsigned(s.count)),f64_from_unsigned(other.count)),f64_from_unsigned(total));
  s.m2=f64_add(s.m2,f64_add(other.m2,correction));
  s.count=total;
  return s;
}
inline ulong variance_finish(VarianceState s, ulong ddof, bool standard) {
  ulong value=f64_div(s.m2,f64_from_unsigned(s.count-ddof));
  return standard?f64_sqrt(value):value;
}
// Translated sliding Welford updates, matching window._moments_bounds.
// The CPU contracts its m2 multiply/add and multiply/subtract to binary64 FMA.
// Nonfinite values affect validity but leave the finite state recoverable.
struct RollingVarianceState { ulong count, bad, mean, m2; };
inline RollingVarianceState rolling_variance_add(RollingVarianceState s, ulong value, ulong origin) {
  if(f64_abs(value)>=0x7ff0000000000000UL){++s.bad;return s;}
  value=f64_sub(value,origin);++s.count;
  ulong delta=f64_sub(value,s.mean);
  s.mean=f64_add(s.mean,f64_div(delta,f64_from_unsigned(s.count)));
  s.m2=f64_fma(delta,f64_sub(value,s.mean),s.m2);
  return s;
}
inline RollingVarianceState rolling_variance_remove(RollingVarianceState s, ulong value, ulong origin) {
  if(f64_abs(value)>=0x7ff0000000000000UL){--s.bad;return s;}
  value=f64_sub(value,origin);--s.count;
  if(!s.count){s.mean=0UL;s.m2=0UL;return s;}
  ulong delta=f64_sub(value,s.mean);
  s.mean=f64_sub(s.mean,f64_div(delta,f64_from_unsigned(s.count)));
  s.m2=f64_fma(delta^0x8000000000000000UL,f64_sub(value,s.mean),s.m2);
  // Mojo max follows maximumNumber: NaN and negative/zero values clamp to +0.
  if(!f64_less(0UL,s.m2))s.m2=0UL;
  return s;
}
inline ulong rolling_variance_finish(RollingVarianceState s, ulong ddof, bool standard) {
  if(s.bad)return 0x7ff8000000000000UL;
  ulong value=f64_div(f64_less(0UL,s.m2)?s.m2:0UL,f64_from_unsigned(s.count-ddof));
  return standard?f64_sqrt(value):value;
}
struct CovarianceState { ulong count,meanX,meanY,cxy,m2x,m2y; };
inline CovarianceState covariance_add(CovarianceState s, ulong x, ulong y) {
  ++s.count;
  ulong n=f64_from_unsigned(s.count);
  ulong dx=f64_sub(x,s.meanX);
  s.meanX=f64_add(s.meanX,f64_div(dx,n));
  ulong dy=f64_sub(y,s.meanY);
  s.meanY=f64_add(s.meanY,f64_div(dy,n));
  s.cxy=f64_add(s.cxy,f64_mul(dx,f64_sub(y,s.meanY)));
  s.m2x=f64_add(s.m2x,f64_mul(dx,f64_sub(x,s.meanX)));
  s.m2y=f64_add(s.m2y,f64_mul(dy,f64_sub(y,s.meanY)));
  return s;
}
inline CovarianceState covariance_merge(CovarianceState s, CovarianceState other) {
  if(!other.count)return s;
  if(!s.count)return other;
  ulong a=f64_from_unsigned(s.count),b=f64_from_unsigned(other.count),n=f64_add(a,b);
  ulong dx=f64_sub(other.meanX,s.meanX),dy=f64_sub(other.meanY,s.meanY);
  s.cxy=f64_add(s.cxy,f64_add(other.cxy,f64_div(f64_mul(f64_mul(f64_mul(dx,dy),a),b),n)));
  s.m2x=f64_add(s.m2x,f64_add(other.m2x,f64_div(f64_mul(f64_mul(f64_mul(dx,dx),a),b),n)));
  s.m2y=f64_add(s.m2y,f64_add(other.m2y,f64_div(f64_mul(f64_mul(f64_mul(dy,dy),a),b),n)));
  s.meanX=f64_add(s.meanX,f64_div(f64_mul(dx,b),n));
  s.meanY=f64_add(s.meanY,f64_div(f64_mul(dy,b),n));
  s.count+=other.count;
  return s;
}
inline ulong covariance_finish(CovarianceState s, ulong ddof, bool correlation) {
  if(correlation){
    if(s.count<2)return f64_div(0UL,0UL);
    return f64_div(s.cxy,f64_sqrt(f64_mul(s.m2x,s.m2y)));
  }
  return s.count<=ddof?0UL:f64_div(s.cxy,f64_from_unsigned(s.count-ddof));
}
)metal";
#endif
