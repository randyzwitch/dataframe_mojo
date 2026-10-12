// Stable partition ordering groups the rows without changing the table. A
// partition owner walks its rows sequentially, preserving CPU floating-point
// association and validating every integer/decimal prefix before head/filter.
std::string cumulativeSource(const DFMRequest &r, const Layout &layout, int64_t i) {
  auto &step = r.steps[i];
  auto h = r.code + 4 * step.start;
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
