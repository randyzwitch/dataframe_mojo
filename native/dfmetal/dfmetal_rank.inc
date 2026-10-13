std::string partitionSource(const DFMRequest &r, const Layout &layout, int64_t i) {
  auto &step = r.steps[i];
  auto name = std::to_string(i);
  auto equal = [&](int64_t slot) {
    auto base = std::to_string(layout.physical[slot]) + "UL*meta.z";
    auto type = slotType(r, slot);
    std::string words = "ulong aa=x[ai],bb=x[bi];";
    if (type == DFM_DECIMAL128) words += "if(x[ai+meta.z]!=x[bi+meta.z])return false;";
    if (type == DFM_FLOAT64_STORAGE)
      words += "aa=!f64_abs(aa)?0UL:f64_nan(aa)?0x7ff8000000000000UL:aa;"
               "bb=!f64_abs(bb)?0UL:f64_nan(bb)?0x7ff8000000000000UL:bb;";
    if (type == DFM_FLOAT32)
      words += "aa=!(aa&0x7fffffffUL)?0UL:(aa&0x7fffffffUL)>0x7f800000UL?0x7fc00000UL:aa;"
               "bb=!(bb&0x7fffffffUL)?0UL:(bb&0x7fffffffUL)>0x7f800000UL?0x7fc00000UL:bb;";
    return "{ulong ai=" + base + "+a,bi=" + base +
        "+b;if(v[ai]!=v[bi])return false;if(v[ai]){" + words +
        "if(aa!=bb)return false;}}";
  };
  std::string text = "bool window_partition" + name +
      "(uint a,uint b,device const T *x,device const uchar *v,constant uint4 &meta){";
  for (int64_t j = 1; j < step.nodes; ++j)
    text += equal(r.code[4 * (step.start + j) + 1]);
  text += "return true;}\n";
  return text;
}

// Rank reuses the stable sort permutation and the otherwise idle order buffer.
// Dense ranks use one block scan and one block-offset pass; no row data or
// permutation is downloaded to the host.
std::string rankSource(const DFMRequest &r, const Layout &layout, int64_t i) {
  auto &step = r.steps[i];
  auto h = r.code + 4 * step.start;
  auto name = std::to_string(i);
  auto output = std::to_string(layout.physical[step.slot]) + "UL*meta.z";
  auto input = std::to_string(layout.physical[h[1]]) + "UL*meta.z";
  auto equal = [&](int64_t slot) {
    auto base = std::to_string(layout.physical[slot]) + "UL*meta.z";
    auto type = slotType(r, slot);
    std::string words = "ulong aa=x[ai],bb=x[bi];";
    if (type == DFM_DECIMAL128) words += "if(x[ai+meta.z]!=x[bi+meta.z])return false;";
    if (type == DFM_FLOAT64_STORAGE)
      words += "aa=!f64_abs(aa)?0UL:f64_nan(aa)?0x7ff8000000000000UL:aa;"
               "bb=!f64_abs(bb)?0UL:f64_nan(bb)?0x7ff8000000000000UL:bb;";
    if (type == DFM_FLOAT32)
      words += "aa=!(aa&0x7fffffffUL)?0UL:(aa&0x7fffffffUL)>0x7f800000UL?0x7fc00000UL:aa;"
               "bb=!(bb&0x7fffffffUL)?0UL:(bb&0x7fffffffUL)>0x7f800000UL?0x7fc00000UL:bb;";
    return "{ulong ai=" + base + "+a,bi=" + base +
        "+b;if(v[ai]!=v[bi])return false;if(v[ai]){" + words +
        "if(aa!=bb)return false;}}";
  };
  std::string text = partitionSource(r, layout, i);
  text += "bool rank_equal" + name +
      "(uint a,uint b,device const T *x,device const uchar *v,constant uint4 &meta){"
      "if(!window_partition" + name + "(a,b,x,v,meta))return false;" +
      equal(h[1]) + "return true;}\n";
  if (h[3] == 3) {
    text += "kernel void rank_prepare" + name + R"MSL(
(device const T *x[[buffer(0)]],device const uchar *v[[buffer(1)]],
 device const uint *order[[buffer(2)]],device T *prefix[[buffer(3)]],
 device uint *blocks[[buffer(4)]],constant uint4 &meta[[buffer(5)]],
 uint row[[thread_position_in_grid]],uint lane[[thread_index_in_threadgroup]],
 uint group[[threadgroup_position_in_grid]]){
 if(group>=(meta.x+255)/256)return;
 threadgroup uint scan[256];uint head=0;
)MSL";
    text += "if(row<meta.x&&v[" + input + "+order[row]])head=uint(row==0||!rank_equal" +
        name + "(order[row-1],order[row],x,v,meta));";
    text += R"MSL(
 scan[lane]=head;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint d=1;d<256;d*=2){uint previous=lane>=d?scan[lane-d]:0;
  threadgroup_barrier(mem_flags::mem_threadgroup);scan[lane]+=previous;
  threadgroup_barrier(mem_flags::mem_threadgroup);}
)MSL";
    text += "if(row<meta.x)prefix[" + output + "+row]=T(scan[lane]);"
            "if(lane==255)blocks[group]=scan[lane];}\n";
    text += "kernel void rank_offsets" + name +
        "(device uint *blocks[[buffer(0)]],constant uint4 &meta[[buffer(1)]]){"
        "uint total=0;for(uint b=0;b<(meta.x+255)/256;++b){uint n=blocks[b];"
        "blocks[b]=total;total+=n;}}\n";
  }
  text += "kernel void rank_assign" + name + R"MSL(
(device T *x[[buffer(0)]],device uchar *v[[buffer(1)]],
 device const uint *order[[buffer(2)]],device const T *prefix[[buffer(3)]],
 device const uint *blocks[[buffer(4)]],constant uint4 &meta[[buffer(5)]],
 uint position[[thread_position_in_grid]]){
 if(position>=meta.x)return;uint own=order[position];
)MSL";
  text += "ulong out=" + output + "+own;bool valid=v[" + input +
      "+own]!=0;v[out]=uchar(valid);x[out]=T(0);if(!valid)return;"
      "uint lo=0,hi=position;while(lo<hi){uint mid=(lo+hi)/2;"
      "if(window_partition" + name + "(order[mid],own,x,v,meta))hi=mid;else lo=mid+1;}"
      "uint base=lo;";
  if (h[3] == 4)
    text += "x[out]=T(position-base+1);";
  else if (h[3] == 3)
    text += "uint count=uint(prefix[" + output + "+position])+blocks[position/256];"
            "uint before=base?uint(prefix[" + output + "+base-1])+blocks[(base-1)/256]:0;"
            "x[out]=T(count-before);";
  else {
    text += "lo=base;hi=position;while(lo<hi){uint mid=(lo+hi)/2;"
            "if(rank_equal" + name + "(order[mid],own,x,v,meta))hi=mid;else lo=mid+1;}"
            "uint first=lo;lo=position+1;hi=meta.x;while(lo<hi){uint mid=(lo+hi)/2;"
            "if(rank_equal" + name + "(order[mid],own,x,v,meta))lo=mid+1;else hi=mid;}";
    if (h[3] == 1) text += "x[out]=T(first-base+1);";
    else if (h[3] == 2) text += "x[out]=T(lo-base);";
    else text += "x[out]=T(f64_div(f64_from_unsigned(ulong(first-base)+1+ulong(lo-base)),0x4000000000000000UL));";
  }
  return text + "}\n";
}
