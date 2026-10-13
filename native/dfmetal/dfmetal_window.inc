// Shift and fill gather words without performing numeric conversions.
std::string shiftFillSource(const DFMRequest &r, const Layout &layout, int64_t i) {
  auto &step = r.steps[i];
  auto h = r.code + 4 * step.start;
  auto name = std::to_string(i);
  auto input = std::to_string(layout.physical[h[1]]) + "UL*meta.z";
  auto output = std::to_string(layout.physical[step.slot]) + "UL*meta.z";
  bool wide = slotType(r, h[1]) == DFM_DECIMAL128;
  std::string text = partitionSource(r, layout, i);
  text += "kernel void cumulative_assign" + name + R"MSL(
(device T *x[[buffer(0)]],device uchar *v[[buffer(1)]],
 device const uint *order[[buffer(2)]],constant uint4 &meta[[buffer(3)]],
 device atomic_uint *error[[buffer(4)]],device const ulong *lit[[buffer(5)]],
 uint position[[thread_position_in_grid]]){
 if(position>=meta.x)return;uint first=order[position];
)MSL";
  text += "long parameter=as_type<long>(lit[" + std::to_string(step.start) + "]);";
  if (h[3] == 4) {
    // Find this row's partition, then gather its shifted source in parallel.
    text += "uint lo=0,hi=position;while(lo<hi){uint mid=(lo+hi)/2;"
            "if(window_partition" + name + "(order[mid],first,x,v,meta))hi=mid;else lo=mid+1;}"
            "uint begin=lo;lo=position+1;hi=meta.x;while(lo<hi){uint mid=(lo+hi)/2;"
            "if(window_partition" + name + "(order[mid],first,x,v,meta))lo=mid+1;else hi=mid;}"
            "uint end=lo,from=0;bool present=false;"
            "if(parameter>=0){ulong offset=ulong(parameter);if(offset<=ulong(position-begin)){from=position-uint(offset);present=true;}}"
            "else{ulong offset=0UL-ulong(parameter);if(offset<ulong(end-position)){from=position+uint(offset);present=true;}}"
            "ulong out=" + output + "+first;x[out]=T(0);v[out]=0;";
    if (wide) text += "x[out+meta.z]=T(0);v[out+meta.z]=0;";
    text += "if(present){uint source=order[from];ulong at=" + input + "+source;"
            "x[out]=x[at];v[out]=v[at];";
    if (wide) text += "x[out+meta.z]=x[at+meta.z];v[out+meta.z]=v[at+meta.z];";
    return text + "}}\n";
  }
  // One owner per partition tracks the nearest original non-null row.
  text += "if(position&&window_partition" + name + "(order[position-1],first,x,v,meta))return;"
          "uint lo=position+1,hi=meta.x;while(lo<hi){uint mid=(lo+hi)/2;"
          "if(window_partition" + name + "(order[mid],first,x,v,meta))lo=mid+1;else hi=mid;}"
          "uint end=lo,last=0,distance=0;bool seen=false;"
          "for(uint offset=0;offset<end-position;++offset){uint row=order[" +
          std::string(h[3] == 6 ? "end-1-offset" : "position+offset") + "];"
          "ulong at=" + input + "+row,out=" + output + "+row;bool valid=v[at]!=0;"
          "if(valid){last=row;seen=true;distance=0;}else ++distance;"
          "bool present=valid||(seen&&(parameter<0||ulong(distance)<=ulong(parameter)));"
          "x[out]=T(0);v[out]=0;";
  if (wide) text += "x[out+meta.z]=T(0);v[out+meta.z]=0;";
  text += "if(present){ulong source=" + input + "+last;x[out]=x[source];v[out]=v[source];";
  if (wide) text += "x[out+meta.z]=x[source+meta.z];v[out+meta.z]=v[source+meta.z];";
  return text + "}}}\n";
}

// One owner per stable partition fills each interior null run once. Original
// values remain anchors, including NaNs and infinities; nearest copies words.
std::string interpolationSource(const DFMRequest &r, const Layout &layout, int64_t i) {
  auto &step = r.steps[i];
  auto h = r.code + 4 * step.start;
  auto name = std::to_string(i);
  auto input = std::to_string(layout.physical[h[1]]) + "UL*meta.z";
  auto output = std::to_string(layout.physical[step.slot]) + "UL*meta.z";
  bool wide = slotType(r, h[1]) == DFM_DECIMAL128;
  bool coordinate = h[3] == 15;
  auto axis = coordinate ? std::to_string(layout.physical[h[2]]) + "UL*meta.z" : "";
  std::string text = partitionSource(r, layout, i);
  text += "kernel void cumulative_assign" + name + R"MSL(
(device T *x[[buffer(0)]],device uchar *v[[buffer(1)]],
 device const uint *order[[buffer(2)]],constant uint4 &meta[[buffer(3)]],
 device atomic_uint *error[[buffer(4)]],device const ulong *lit[[buffer(5)]],
 uint position[[thread_position_in_grid]]){
 if(position>=meta.x)return;uint first=order[position];
)MSL";
  text += "if(position&&window_partition" + name + "(order[position-1],first,x,v,meta))return;"
          "uint lo=position+1,hi=meta.x;while(lo<hi){uint mid=(lo+hi)/2;"
          "if(window_partition" + name + "(order[mid],first,x,v,meta))lo=mid+1;else hi=mid;}"
          "uint end=lo;for(uint j=position;j<end;++j){uint row=order[j];"
          "ulong at=" + input + "+row,out=" + output + "+row;"
          "x[out]=v[at]?x[at]:T(0);v[out]=v[at];";
  if (wide) text += "x[out+meta.z]=v[at]?x[at+meta.z]:T(0);v[out+meta.z]=v[at];";
  text += "}uint previous=position;bool seen=false;"
          "for(uint j=position;j<end;++j){ulong at=" + input + "+order[j];"
          "if(v[at]){" + std::string(coordinate ? "if(v[" + axis + "+order[j]])" : "") +
          "{previous=j;seen=true;}continue;}if(!seen)continue;"
          "uint following=j+1;while(following<end&&(!v[" + input + "+order[following]]" +
          (coordinate ? "||!v[" + axis + "+order[following]]" : "") + "))++following;"
          "if(following==end)break;ulong before=" + input + "+order[previous],after=" + input + "+order[following];"
          "for(uint p=j;p<following;++p){ulong out=" + output + "+order[p];";
  if (h[3] == 14) {
    text += "ulong source=p-previous<following-p?before:after;x[out]=x[source];";
    if (wide) text += "x[out+meta.z]=x[source+meta.z];v[out+meta.z]=1;";
  } else {
    if (coordinate) {
      text += "ulong point=" + axis + "+order[p],left=" + axis + "+order[previous],right=" + axis + "+order[following];"
              "if(v[" + input + "+order[p]]||!v[point])continue;"
              "ulong span=f64_sub(x[right],x[left]);if(!f64_abs(span))continue;"
              "ulong fraction=f64_div(f64_sub(x[point],x[left]),span);";
    } else
      text += "ulong fraction=f64_div(f64_from_unsigned(ulong(p-previous)),f64_from_unsigned(ulong(following-previous)));";
    text += "x[out]=T(f64_fma(f64_sub(x[after],x[before]),fraction,x[before]));";
  }
  return text + "v[out]=1;}j=following-1;}}\n";
}

// Each partition owner follows the CPU's monotone sliding bounds. Every
// input is added/removed once, and output rows retain source order.
std::string rollingMomentsSource(const DFMRequest &r, const Layout &layout, int64_t i) {
  auto &step = r.steps[i];
  auto h = r.code + 4 * step.start;
  auto name = std::to_string(i);
  auto input = std::to_string(layout.physical[h[1]]) + "UL*meta.z";
  auto output = std::to_string(layout.physical[step.slot]) + "UL*meta.z";
  std::string text = partitionSource(r, layout, i);
  text += "kernel void cumulative_assign" + name + R"MSL(
(device T *x[[buffer(0)]],device uchar *v[[buffer(1)]],
 device const uint *order[[buffer(2)]],constant uint4 &meta[[buffer(3)]],
 device atomic_uint *error[[buffer(4)]],device const ulong *lit[[buffer(5)]],
 uint position[[thread_position_in_grid]]){
 if(position>=meta.x)return;uint first=order[position];
)MSL";
  text += "if(position&&window_partition" + name + "(order[position-1],first,x,v,meta))return;"
          "uint lo=position+1,hi=meta.x;while(lo<hi){uint mid=(lo+hi)/2;"
          "if(window_partition" + name + "(order[mid],first,x,v,meta))lo=mid+1;else hi=mid;}"
          "uint end=lo;ulong origin=0UL;"
          "for(uint j=position;j<end;++j){ulong at=" + input + "+order[j];"
          "if(v[at]&&f64_abs(x[at])<0x7ff0000000000000UL){origin=x[at];break;}}"
          "ulong parameter=lit[" + std::to_string(step.start) + "];"
          "uint window=uint(parameter);ulong ddof=parameter>>32;"
          "lo=position;hi=position;RollingVarianceState state={0UL,0UL,0UL,0UL};"
          "for(uint j=position;j<end;++j){uint start=j-position+1>window?j-window+1:position;"
          "bool rebuild=start<lo||start>=hi;if(rebuild){lo=start;hi=start;state={0UL,0UL,0UL,0UL};}"
          "while(lo<start){ulong at=" + input + "+order[lo++];"
          "if(v[at]){ulong previous=state.m2;state=rolling_variance_remove(state,x[at],origin);"
          "rebuild=rebuild||(f64_abs(x[at])<0x7ff0000000000000UL&&(!state.count||f64_abs(previous)>=0x7ff0000000000000UL||f64_less(state.m2,f64_mul(previous,0x3fe0000000000000UL))));}}"
          "if(rebuild){lo=start;hi=start;state={0UL,0UL,0UL,0UL};"
          "for(uint p=start;p<=j;++p){ulong at=" + input + "+order[p];if(v[at]&&f64_abs(x[at])<0x7ff0000000000000UL){origin=x[at];break;}}}"
          "while(hi<=j){ulong at=" + input + "+order[hi++];"
          "if(v[at])state=rolling_variance_add(state,x[at],origin);}"
          "ulong out=" + output + "+order[j];x[out]=T(0);v[out]=0;"
          "ulong total=state.count+state.bad;if(total>ddof&&total>=" +
          std::to_string(h[2]) + "UL){x[out]=T(rolling_variance_finish(state,ddof," +
          std::string(h[3] == 12 ? "true" : "false") + "));v[out]=1;}}}";
  return text + "\n";
}

// Recompute each trailing window in source order, matching CPU association.
std::string rollingSource(const DFMRequest &r, const Layout &layout, int64_t i) {
  auto &step = r.steps[i];
  auto h = r.code + 4 * step.start;
  auto name = std::to_string(i);
  auto inputType = slotType(r, h[1]), outputType = slotType(r, step.slot);
  auto input = std::to_string(layout.physical[h[1]]) + "UL*meta.z";
  auto output = std::to_string(layout.physical[step.slot]) + "UL*meta.z";
  bool arithmetic = h[3] == 7 || h[3] == 8;
  bool integer = h[3] == 7 && inputType != DFM_FLOAT32 && inputType != DFM_FLOAT64_STORAGE;
  bool wide = inputType == DFM_DECIMAL128;
  std::string text = partitionSource(r, layout, i);
  text += "kernel void cumulative_assign" + name + R"MSL(
(device T *x[[buffer(0)]],device uchar *v[[buffer(1)]],
 device const uint *order[[buffer(2)]],constant uint4 &meta[[buffer(3)]],
 device atomic_uint *error[[buffer(4)]],device const ulong *lit[[buffer(5)]],
 uint position[[thread_position_in_grid]]){
 if(position>=meta.x)return;uint own=order[position];
)MSL";
  if (arithmetic && inputType == DFM_UINT64)
    text += "{ulong at=" + input + "+own;if(v[at]&&x[at]>0x7fffffffffffffffUL){"
            "atomic_fetch_min_explicit(error,0xD0000000U,memory_order_relaxed);return;}}";
  text += "uint lo=0,hi=position;while(lo<hi){uint mid=(lo+hi)/2;"
          "if(window_partition" + name + "(order[mid],own,x,v,meta))hi=mid;else lo=mid+1;}"
          "uint begin=lo;ulong window=lit[" + std::to_string(step.start) + "];"
          "uint start=ulong(position-begin+1)>window?position-uint(window)+1:begin;"
          "ulong state=0UL,upper=0UL;uint count=0;"
          "for(uint atPosition=start;atPosition<=position;++atPosition){uint row=order[atPosition];"
          "ulong at=" + input + "+row;if(!v[at])continue;ulong word=x[at];";
  if (integer)
    text += "ulong other=long(word)<0?~0UL:0UL;ulong next=state+word;"
            "upper+=other+ulong(next<state);state=next;";
  else if (arithmetic) {
    auto word = inputType == DFM_FLOAT32 ? "f64_from_float32(uint(word))" :
        inputType == DFM_FLOAT64_STORAGE ? "word" : "f64_from_signed(long(word))";
    text += "state=f64_add(state," + std::string(word) + ");";
  } else {
    if (wide)
      text += "ulong other=x[at+meta.z];if(!count||" +
          std::string(h[3] == 10 ? "wide_less(ulong2(state,upper),ulong2(word,other))" :
                                  "wide_less(ulong2(word,other),ulong2(state,upper))") +
          "){state=word;upper=other;}";
    else
      text += "if(!count||extrema_better(word,state," + std::to_string(inputType) + "," +
          std::string(h[3] == 10 ? "true" : "false") + "))state=word;";
  }
  text += "++count;}ulong out=" + output + "+own;x[out]=T(0);v[out]=0;";
  if (wide) text += "x[out+meta.z]=T(0);v[out+meta.z]=0;";
  text += "if(!count||ulong(count)<" + std::to_string(h[2]) + "UL)return;";
  if (integer) {
    text += "bool overflow=upper!=(long(state)<0?~0UL:0UL);";
    if (outputType == DFM_INT32)
      text += "overflow=overflow||long(state)<(-2147483647L-1L)||long(state)>2147483647L;";
    if (outputType == DFM_UINT32)
      text += "overflow=overflow||long(state)<0L||state>4294967295UL;";
    if (outputType == DFM_UINT64)
      text += "overflow=overflow||long(state)<0L;";
    text += "if(overflow){atomic_fetch_min_explicit(error,0xD0000000U,memory_order_relaxed);return;}";
  }
  if (h[3] == 8) text += "state=f64_div(state,f64_from_unsigned(ulong(count)));";
  if (h[3] == 7 && inputType == DFM_FLOAT32) text += "state=ulong(f64_to_float32(state));";
  text += "x[out]=T(state);v[out]=1;";
  if (wide) text += "x[out+meta.z]=T(upper);v[out+meta.z]=1;";
  return text + "}\n";
}

// Duration bounds use the original row order within stable partitions. Each
// owner validates its whole index before evaluating a single window.
std::string durationSource(const DFMRequest &r, const Layout &layout, int64_t i) {
  auto &step=r.steps[i];auto h=r.code+4*step.start;auto bounds=h+4;
  auto name=std::to_string(i);
  auto input=std::to_string(layout.physical[h[1]])+"UL*meta.z";
  auto axis=std::to_string(layout.physical[h[2]])+"UL*meta.z";
  auto output=std::to_string(layout.physical[step.slot])+"UL*meta.z";
  auto inputType=slotType(r,h[1]),outputType=slotType(r,step.slot);
  bool moments=h[3]>=4,arithmetic=h[3]<=1;
  bool decimal=h[3]==0&&inputType==DFM_DECIMAL128;
  bool integer=h[3]==0&&inputType!=DFM_FLOAT32&&inputType!=DFM_FLOAT64_STORAGE&&!decimal;
  bool wide=inputType==DFM_DECIMAL128;
  bool floating=inputType==DFM_FLOAT32||inputType==DFM_FLOAT64_STORAGE;
  std::string text=partitionSource(r,layout,i);
  text+="kernel void cumulative_assign"+name+R"MSL(
(device T *x[[buffer(0)]],device uchar *v[[buffer(1)]],
 device const uint *order[[buffer(2)]],constant uint4 &meta[[buffer(3)]],
 device atomic_uint *error[[buffer(4)]],device const ulong *lit[[buffer(5)]],
 uint position[[thread_position_in_grid]]){
 if(position>=meta.x)return;uint first=order[position];
)MSL";
  text+="if(position&&window_partition"+name+"(order[position-1],first,x,v,meta))return;"
        "uint lo=position+1,hi=meta.x;while(lo<hi){uint mid=(lo+hi)/2;"
        "if(window_partition"+name+"(order[mid],first,x,v,meta))lo=mid+1;else hi=mid;}uint end=lo;"
        "long period=as_type<long>(lit["+std::to_string(step.start)+"]);"
        "for(uint j=position;j<end;++j){ulong at="+axis+"+order[j];"
        "if(!v[at]){atomic_fetch_min_explicit(error,0xB0000000U,memory_order_relaxed);return;}"
        "if(j>position&&as_type<long>(x[at])<as_type<long>(x["+axis+"+order[j-1]])){"
        "atomic_fetch_min_explicit(error,0xB0000001U,memory_order_relaxed);return;}}";
  if(moments)
    text+="ulong origin=0UL;for(uint j=position;j<end;++j){ulong at="+input+"+order[j];"
          "if(v[at]&&f64_abs(x[at])<0x7ff0000000000000UL){origin=x[at];break;}}"
          "RollingVarianceState state={0UL,0UL,0UL,0UL};uint previousLower=position,previousUpper=position;";
  text+="for(uint j=position;j<end;++j){uint own=order[j];long time=as_type<long>(x["+axis+"+own]);"
        "if(time<(-9223372036854775807L-1L)+period){atomic_fetch_min_explicit(error,0xB0000002U,memory_order_relaxed);return;}"
        "long startTime=time-period;lo=position;hi=end;while(lo<hi){uint mid=(lo+hi)/2;long t=as_type<long>(x["+axis+"+order[mid]]);"
        "if(t<startTime"+std::string(bounds[2]&1?"":"||t==startTime")+")lo=mid+1;else hi=mid;}uint start=lo;"
        "lo=position;hi=end;while(lo<hi){uint mid=(lo+hi)/2;long t=as_type<long>(x["+axis+"+order[mid]]);"
        "if(t<time"+std::string(bounds[2]&2?"||t==time":"")+")lo=mid+1;else hi=mid;}uint stop=max(start,lo);"
        "ulong out="+output+"+own;x[out]=T(0);v[out]=0;";
  if(wide)text+="x[out+meta.z]=T(0);v[out+meta.z]=0;";
  if(moments){
    text+="bool rebuild=start<previousLower||stop<previousUpper||start>=previousUpper;if(rebuild){previousLower=start;previousUpper=start;state={0UL,0UL,0UL,0UL};}"
          "while(previousLower<start){ulong at="+input+"+order[previousLower++];if(v[at]){ulong previous=state.m2;state=rolling_variance_remove(state,x[at],origin);"
          "rebuild=rebuild||(f64_abs(x[at])<0x7ff0000000000000UL&&(!state.count||f64_abs(previous)>=0x7ff0000000000000UL||f64_less(state.m2,f64_mul(previous,0x3fe0000000000000UL))));}}"
          "if(rebuild){previousLower=start;previousUpper=start;state={0UL,0UL,0UL,0UL};"
          "for(uint p=start;p<stop;++p){ulong at="+input+"+order[p];if(v[at]&&f64_abs(x[at])<0x7ff0000000000000UL){origin=x[at];break;}}}"
          "while(previousUpper<stop){ulong at="+input+"+order[previousUpper++];if(v[at])state=rolling_variance_add(state,x[at],origin);}"
          "ulong total=state.count+state.bad;if(total>="+std::to_string(bounds[1])+"UL&&total>"+std::to_string(bounds[3])+"UL){"
          "x[out]=T(rolling_variance_finish(state,"+std::to_string(bounds[3])+"UL,"+std::string(h[3]==5?"true":"false")+"));v[out]=1;}";
    return text+"}}\n";
  }
  if(decimal)
    text+="uint samples=0;for(uint p=start;p<stop;++p) samples+=uint(v["+input+"+order[p]]!=0);"
          "if(ulong(samples)<"+std::to_string(bounds[1])+"UL)continue;"
          "ulong2 limit=decimal_wide_pow10("+std::to_string(bounds[3])+"U);";
  text+="ulong state=0UL,upper=0UL;uint count=0;for(uint p=start;p<stop;++p){ulong at="+input+"+order[p];"
        "if(!v[at])continue;ulong word=x[at];";
  if(decimal)
    text+="ulong2 value=ulong2(word,x[at+meta.z]),previous=ulong2(state,upper);"
          "ulong2 next=decimal_wide_add(previous,value);"
          "bool overflow=((previous.y^value.y)&0x8000000000000000UL)==0UL&&((previous.y^next.y)&0x8000000000000000UL)!=0UL;"
          "if(overflow||decimal_wide_ge(decimal_magnitude(value),limit)||decimal_wide_ge(decimal_magnitude(next),limit)){"
          "atomic_fetch_min_explicit(error,0xB0000003U,memory_order_relaxed);return;}state=next.x;upper=next.y;";
  else if(integer)
    text+="ulong other="+std::string(unsignedType(inputType)?"0UL":"long(word)<0?~0UL:0UL")+";ulong next=state+word;upper+=other+ulong(next<state);state=next;";
  else if(arithmetic)
    text+="state=f64_add(state,"+std::string(inputType==DFM_FLOAT32?"f64_from_float32(uint(word))":"word")+");";
  else if(wide)
    text+="ulong other=x[at+meta.z];if(!count||"+std::string(h[3]==3?"wide_less(ulong2(state,upper),ulong2(word,other))":"wide_less(ulong2(word,other),ulong2(state,upper))")+"){state=word;upper=other;}";
  else {
    std::string better="extrema_better(word,state,"+std::to_string(inputType)+","+std::string(h[3]==3?"true":"false")+")";
    if(floating)
      better="f64_nan(f64_word(word,"+std::to_string(inputType)+"))||(!f64_nan(f64_word(state,"+std::to_string(inputType)+"))&&"+better+")";
    text+="if(!count||"+better+")state=word;";
  }
  text+="++count;}if(ulong(count)<"+std::to_string(bounds[1])+"UL"+std::string(h[3]==0?"":"||!count")+")continue;";
  if(integer){
    text+="bool overflow="+std::string(unsignedType(outputType)?"upper!=0UL":"upper!=(long(state)<0?~0UL:0UL)")+";";
    if(outputType==DFM_INT32)text+="overflow=overflow||long(state)<(-2147483647L-1L)||long(state)>2147483647L;";
    if(outputType==DFM_UINT32)text+="overflow=overflow||state>4294967295UL;";
    text+="if(overflow){atomic_fetch_min_explicit(error,0xD0000000U,memory_order_relaxed);return;}";
  }
  if(h[3]==1)text+="state=f64_div(state,f64_from_unsigned(ulong(count)));";
  if(h[3]==0&&inputType==DFM_FLOAT32)text+="state=ulong(f64_to_float32(state));";
  text+="x[out]=T(state);v[out]=1;";
  if(wide)text+="x[out+meta.z]=T(upper);v[out+meta.z]=1;";
  return text+"}}\n";
}

// Stable partition ordering groups the rows without changing the table. A
// partition owner walks its rows sequentially, preserving CPU floating-point
// association and validating every integer/decimal prefix before head/filter.
std::string cumulativeSource(const DFMRequest &r, const Layout &layout, int64_t i) {
  auto &step = r.steps[i];
  auto h = r.code + 4 * step.start;
  if (h[0] == DFM_DURATION_KEY) return durationSource(r,layout,i);
  if (h[3] >= 13) return interpolationSource(r, layout, i);
  if (h[3] >= 11) return rollingMomentsSource(r, layout, i);
  if (h[3] >= 7) return rollingSource(r, layout, i);
  if (h[3] >= 4) return shiftFillSource(r, layout, i);
  auto name = std::to_string(i);
  auto inputType = slotType(r, h[1]), outputType = slotType(r, step.slot);
  auto input = std::to_string(layout.physical[h[1]]) + "UL*meta.z";
  auto output = std::to_string(layout.physical[step.slot]) + "UL*meta.z";
  std::string text = partitionSource(r, layout, i);
  text += "kernel void cumulative_assign" + name + R"MSL(
(device T *x[[buffer(0)]],device uchar *v[[buffer(1)]],
 device const uint *order[[buffer(2)]],constant uint4 &meta[[buffer(3)]],
 device atomic_uint *error[[buffer(4)]],uint position[[thread_position_in_grid]]){
 if(position>=meta.x)return;uint first=order[position];
)MSL";
  text += "if(position&&window_partition" + name + "(order[position-1],first,x,v,meta))return;"
          "uint lo=position+1,hi=meta.x;while(lo<hi){uint mid=(lo+hi)/2;"
          "if(window_partition" + name + "(order[mid],first,x,v,meta))lo=mid+1;else hi=mid;}"
          "uint end=lo;ulong state=0UL,upper=0UL;bool seen=false;"
          "for(uint offset=0;offset<end-position;++offset){uint row=order[" +
          std::string(h[2] ? "end-1-offset" : "position+offset") + "];"
          "ulong at=" + input + "+row,out=" + output + "+row;bool valid=v[at]!=0;"
          "ulong word=x[at];";
  if (h[3] == 3) {
    text += "state+=ulong(valid);x[out]=T(state);v[out]=1;";
  } else {
    text += "x[out]=T(0);v[out]=uchar(valid);";
    if (outputType == DFM_DECIMAL128)
      text += "x[out+meta.z]=T(0);v[out+meta.z]=uchar(valid);";
    text += "if(!valid)continue;";
    if (h[3] == 0) {
      if (outputType == DFM_DECIMAL128) {
        // Generate the exact declared precision limit without host Int128.
        uint64_t low = 0, high = 0;
        for (int64_t digit = 0; digit < int64_t(r.literals[step.start]); ++digit) {
          // 10*n+9, with n <= 10**37-1 before the final iteration.
          // Use four 32-bit limbs to avoid compiler-specific 128-bit types.
          uint64_t limb = (low & 0xffffffffULL) * 10 + 9;
          uint64_t top = (low >> 32) * 10 + (limb >> 32);
          uint64_t next = (top << 32) | uint32_t(limb);
          high = high * 10 + (top >> 32);
          low = next;
        }
        text += "ulong other=" + std::string(inputType == DFM_DECIMAL128 ? "x[at+meta.z]" : "(long(word)<0?~0UL:0UL)") + ";"
                "ulong next=state+word,nextUpper=upper+other+ulong(next<state);"
                "bool overflow=((upper^other)&0x8000000000000000UL)==0UL&&"
                "((upper^nextUpper)&0x8000000000000000UL)!=0UL;"
                "bool negative=long(nextUpper)<0;ulong magnitude=negative?0UL-next:next;"
                "ulong magnitudeUpper=negative?~nextUpper+ulong(magnitude==0UL):nextUpper;"
                "if(overflow||magnitudeUpper>" + std::to_string(high) + "UL||"
                "(magnitudeUpper==" + std::to_string(high) + "UL&&magnitude>" + std::to_string(low) + "UL)){"
                "atomic_fetch_min_explicit(error,0xD0000000U,memory_order_relaxed);return;}"
                "state=next;upper=nextUpper;";
      } else if (inputType == DFM_FLOAT64_STORAGE || inputType == DFM_FLOAT32) {
        text += "state=f64_add(state," + std::string(inputType == DFM_FLOAT32 ? "f64_from_float32(uint(word))" : "word") + ");";
      } else {
        if (inputType == DFM_UINT64)
          text += "if(word>0x7fffffffffffffffUL){atomic_fetch_min_explicit(error,0xD0000000U,memory_order_relaxed);return;}";
        text += "ulong next=state+word;bool overflow=((state^word)&0x8000000000000000UL)==0UL&&"
                "((state^next)&0x8000000000000000UL)!=0UL;";
        if (outputType == DFM_INT32)
          text += "overflow=overflow||long(next)<(-2147483647L-1L)||long(next)>2147483647L;";
        if (outputType == DFM_UINT32)
          text += "overflow=overflow||long(next)<0L||next>4294967295UL;";
        text += "if(overflow){atomic_fetch_min_explicit(error,0xD0000000U,memory_order_relaxed);return;}state=next;";
      }
    } else {
      std::string better;
      if (inputType == DFM_DECIMAL128) {
        text += "ulong other=x[at+meta.z];";
        better = h[3] == 2 ? "wide_less(ulong2(state,upper),ulong2(word,other))" : "wide_less(ulong2(word,other),ulong2(state,upper))";
      } else
        better = "extrema_better(word,state," + std::to_string(inputType) + "," + (h[3] == 2 ? "true" : "false") + ")";
      text += "if(!seen||" + better + "){state=word;" + (inputType == DFM_DECIMAL128 ? "upper=other;" : "") + "}seen=true;";
    }
    text += "x[out]=T(" + std::string(h[3] == 0 && inputType == DFM_FLOAT32 ? "f64_to_float32(state)" : "state") + ");";
    if (outputType == DFM_DECIMAL128) text += "x[out+meta.z]=T(upper);";
  }
  return text + "}}\n";
}
