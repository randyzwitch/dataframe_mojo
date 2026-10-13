#pragma once

// Fixed-pass binary64 order statistics. Each byte refines the selected radix
// prefix; eight scans find a rank without a row-sized scratch allocation.
static constexpr const char *dfm_quantiles_source = R"metal(
inline ulong quantile_key(ulong word) {
  if(f64_nan(word))return ~0UL;
  if(!f64_abs(word))return 0x8000000000000000UL;
  return word>>63?~word:word^0x8000000000000000UL;
}
template<typename Q> uint quantile_row(device const Q *x,ulong index,uint at){
  return index==~0UL?at:uint(x[index+at]);
}
template<typename Q> ulong quantile_select(device const Q *x,device const uchar *v,
 ulong base,ulong index,uint begin,uint end,uint rank){
  ulong prefix=0,mask=0;
  for(uint pass=0;pass<8;++pass){
    uint bins[256];for(uint bin=0;bin<256;++bin)bins[bin]=0;
    uint shift=56U-8U*pass;
    for(uint at=begin;at<end;++at){ulong from=base+quantile_row(x,index,at);
      if(!v[from])continue;ulong key=quantile_key(ulong(x[from]));
      if((key&mask)==prefix)++bins[uint((key>>shift)&255UL)];}
    uint chosen=0;while(chosen<255&&rank>=bins[chosen]){rank-=bins[chosen];++chosen;}
    prefix|=ulong(chosen)<<shift;mask|=255UL<<shift;
  }
  if(prefix==~0UL)return f64_invalid();
  return prefix>>63?prefix^0x8000000000000000UL:~prefix;
}
template<typename Q> ulong quantile_reduce(device const Q *x,device const uchar *v,
 ulong base,ulong index,uint begin,uint end,ulong q,int kind,thread bool &valid){
  uint n=0;for(uint at=begin;at<end;++at)n+=uint(v[base+quantile_row(x,index,at)]!=0);
  valid=n!=0;if(!valid)return 0UL;
  ulong position=f64_mul(q,f64_from_unsigned(ulong(n-1U)));
  uint lower=uint(f64_to_integer(position,32U,true).word);
  ulong fraction=f64_sub(position,f64_from_unsigned(lower));
  uint upper=min(lower+uint(f64_abs(fraction)!=0UL),n-1U);
  uint wanted=kind==223?upper:kind==224?
    min(uint(f64_to_integer(f64_add(position,0x3fe0000000000000UL),32U,true).word),n-1U):lower;
  ulong lo=quantile_select(x,v,base,index,begin,end,wanted);
  if(kind==222||kind==223||kind==224||lower==upper||!f64_abs(fraction))return lo;
  ulong hi=quantile_select(x,v,base,index,begin,end,upper);
  if(kind==225)return f64_div(f64_add(lo,hi),0x4000000000000000UL);
  return f64_fma(f64_sub(hi,lo),fraction,lo);
}
)metal";
