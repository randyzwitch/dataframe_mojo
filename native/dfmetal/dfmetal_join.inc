// GPU joins use one synchronization to learn cardinality and one to gather.
// Metadata is owned by the session; caller storage is touched only on success.
#include <cstddef>
static_assert(offsetof(DFMJoinInput, side) == 40 &&
              offsetof(DFMJoinOutput, prefer_right) == 40);
static_assert(sizeof(DFMJoinInput) == 48 && sizeof(DFMJoinKey) == 16);
static_assert(sizeof(DFMJoinOutput) == 48 && sizeof(DFMJoinRequest) == 104);
static_assert(offsetof(DFMJoinRequest, inputs) == 80);
namespace {
struct JColumn {
  int64_t word, values, validity, rows, bit, type;
};
struct JOutput {
  // Low bit selects the preferred side; upper bits name the physical output
  // slot. A Decimal128 output reserves two slots without changing this layout.
  int64_t type, left, right, prefer;
};
struct JMeta {
  uint32_t probe, build, side, columns;
  uint64_t total, leftTotal, unmatched, limit;
};
static_assert(sizeof(JColumn) == 48 && sizeof(JOutput) == 32 &&
              sizeof(JMeta) == 48);
struct JShape {
  std::vector<JColumn> columns;
  int64_t raw = 0, words = 0, probe = 0, build = 0, pb = 0, bb = 0;
  DFMMemory memory{};
  bool hash = false;
};
int64_t jpacked(int64_t rows, int64_t offset = 0) {
  return rows ? add(add(rows, offset), 7) / 8 : 0;
}
int64_t jOutputSlots(const DFMJoinRequest &r) {
  int64_t slots = 0;
  for (int64_t i = 0; i < r.output_count; ++i)
    slots = add(slots, r.outputs[i].dtype == DFM_DECIMAL128 ? 2 : 1);
  return slots;
}
JShape jvalidate(const DFMJoinRequest &r, bool pointers = false,
                 const DFMResidentInput *refs = nullptr) {
  if (r.abi_version != DFM_JOIN_ABI_VERSION || r.left_rows < 0 ||
      r.right_rows < 0 || r.left_rows > INT32_MAX || r.right_rows > INT32_MAX ||
      r.input_count < 0 || r.input_count > 64 || r.key_count < 0 ||
      r.key_count > 64 || r.output_count < 1 || r.output_count > 64 ||
      r.how < 0 || r.how > 6 || r.limit < -1 || r.memory_budget_bytes < -1 ||
      (r.profiling != 0 && r.profiling != 1) || (r.input_count && !r.inputs) ||
      (r.key_count && !r.keys) || !r.outputs ||
      ((r.how == 6) != (r.key_count == 0)))
    fail("Invalid Metal join request");
  JShape s;
  s.probe = r.how == 2 ? r.right_rows : r.left_rows;
  s.build = r.how == 2 ? r.left_rows : r.right_rows;
  s.hash = r.how != 6 && r.memory_budget_bytes == -1 && hashCandidate(s.build);
  s.pb = std::max<int64_t>(1, (s.probe + 255) / 256);
  s.bb = std::max<int64_t>(1, (s.build + 255) / 256);
  for (int64_t i = 0; i < r.input_count; ++i) {
    auto &a = r.inputs[i];
    auto &v = a.input;
    if ((a.side != 0 && a.side != 1) || v.bit_offset < 0 || v.bit_offset > 7 ||
        (v.has_validity != 0 && v.has_validity != 1))
      fail("Invalid Metal join input");
    auto rows = a.side ? r.right_rows : r.left_rows;
    auto w = width(v.dtype), bitmap = jpacked(rows, v.bit_offset);
    bool native = refs && refs[i].table;
    if (pointers && !native && rows &&
        (!v.values || (v.has_validity && !v.validity)))
      fail("Missing Metal join input storage");
    s.raw = aligned(s.raw);
    auto at = s.raw;
    auto payload = v.dtype == DFM_BOOL ? bitmap : mul(rows, w);
    if (!native) {
      s.raw = add(s.raw, payload);
      s.memory.upload_bytes = add(s.memory.upload_bytes, payload);
    }
    auto valid = !native && v.has_validity ? s.raw : -1;
    if (!native && v.has_validity) {
      s.raw = add(s.raw, bitmap);
      s.memory.upload_bytes = add(s.memory.upload_bytes, bitmap);
    }
    s.columns.push_back({s.words, native ? -1 : at, valid, rows, v.bit_offset, v.dtype});
    s.words = add(s.words, mul(std::max<int64_t>(1, rows), v.dtype == DFM_DECIMAL128 ? 2 : 1));
  }
  auto input = [&](int64_t index, int64_t side) {
    if (index < 0 || index >= r.input_count || r.inputs[index].side != side)
      fail("Invalid Metal join source index");
  };
  for (int64_t i = 0; i < r.key_count; ++i) {
    auto k = r.keys[i];
    input(k.left, 0);
    input(k.right, 1);
    if (r.inputs[k.left].input.dtype != r.inputs[k.right].input.dtype)
      fail("Metal join key dtype mismatch");
  }
  for (int64_t i = 0; i < r.output_count; ++i) {
    auto o = r.outputs[i];
    width(o.dtype);
    if (o.left < -1 || o.right < -1 || (o.left == -1 && o.right == -1) ||
        (o.prefer_right != 0 && o.prefer_right != 1) ||
        (o.prefer_right && o.right < 0) ||
        ((r.how == 4 || r.how == 5) && o.right >= 0))
      fail("Invalid Metal join output");
    if (o.left >= 0) {
      input(o.left, 0);
      if (r.inputs[o.left].input.dtype != o.dtype)
        fail("Metal join output dtype mismatch");
    }
    if (o.right >= 0) {
      input(o.right, 1);
      if (r.inputs[o.right].input.dtype != o.dtype)
        fail("Metal join output dtype mismatch");
    }
  }
  // Every allocation, including the one-byte empty buffers, is accounted for.
  auto charge = [&](int64_t bytes) {
    s.memory.shared_bytes =
        add(s.memory.shared_bytes, std::max<int64_t>(1, bytes));
  };
  for (auto bytes : {s.raw, mul(s.words, 8), s.words, mul(r.input_count, 48),
                     mul(r.key_count, 16), mul(r.output_count, 32), int64_t(48),
                     mul(s.build, 4), mul(s.build, 4), mul(s.probe, 8),
                     mul(s.probe, 8), mul(s.pb, 8), mul(add(s.build, 1), 4),
                     mul(s.build, 8), mul(s.bb, 8), mul(s.build, 4)})
    charge(bytes);
  if (s.hash) {
    RadixShape radix(s.build);
    charge(mul(radix.cap, 8)); charge(mul(radix.cap, 4));
    charge(radix.histogram); charge(64); charge(32);
    charge(mul(hashCapacity(s.build), 4)); charge(mul(hashCapacity(s.build), 8));
  }
  s.memory.peak_bytes = s.memory.shared_bytes;
  s.memory.blocks = s.pb;
  s.memory.capacity = std::max<int64_t>(1, s.probe);
  s.memory.launches = (r.input_count ? 1 : 0) + (r.how == 6 ? 0 : 1) + 5;
  if (s.hash) {
    auto bits = 0;
    for (auto cap = hashCapacity(s.build); cap; cap >>= 1) ++bits;
    s.memory.launches += 3 + 3 * ((bits + 3) / 4);
  } else if (r.how != 6)
    for (int64_t width = 1; width < s.build; width *= 2)
      ++s.memory.launches;
  if (r.how == 3)
    s.memory.launches += 6;
  return s;
}
std::string jsource(int64_t keys, int64_t how, bool hash) {
  std::string src = "#include <metal_stdlib>\nusing namespace metal;\n";
  src += residentSource();
  src += "constant uint KEYS=" + std::to_string(keys) +
         ";\nconstant uint HOW=" + std::to_string(how) + ";\n";
  src += orderingSource();
  if (hash) src += radixSource();
  src += R"MSL(
struct Col { long word, values, validity, rows, bit, type; };
struct Out { long type, left, right, prefer; };
struct Meta { uint probe, build, side, columns; ulong total, leftTotal, unmatched, limit; };
kernel void j_unpack(device const uchar *raw[[buffer(0)]],device ulong *x[[buffer(1)]],
 device uchar *v[[buffer(2)]],device const Col *cols[[buffer(3)]],constant Meta &m[[buffer(4)]],uint2 at[[thread_position_in_grid]]) {
 if(at.y>=m.columns)return;Col c=cols[at.y];if(at.x>=c.rows||c.values<0)return;
 ulong i=c.word+at.x,b=ulong(at.x)+c.bit;device const uchar *p=raw+c.values;
 ulong z=0;switch(c.type){case 1:case 9:z=((device const uint*)p)[at.x];break;
 case 2:z=ulong(((device const int*)p)[at.x]);break;case 3:case 10:case 11:z=((device const ulong*)p)[at.x];break;
 case 12:z=((device const ulong*)p)[2*at.x];break;
 case 4:z=(p[b/8]>>(b%8))&1;break;case 5:z=ulong(((device const char*)p)[at.x]);break;
 case 6:z=ulong(((device const short*)p)[at.x]);break;case 7:z=p[at.x];break;case 8:z=((device const ushort*)p)[at.x];break;}
 x[i]=z;v[i]=c.validity<0?1:((raw[c.validity+b/8]>>(b%8))&1);
 if(c.type==12){x[i+max(c.rows,1L)]=((device const ulong*)p)[2*at.x+1];v[i+max(c.rows,1L)]=v[i];}
}
// Nulls sort first. Matching separately excludes every tuple containing null.
int j_cmp(uint a,uint b,bool cross,device const ulong *x,device const uchar *v,
 device const Col *cols,device const long2 *keys,constant Meta &m) {
 for(uint k=0;k<KEYS;++k){long2 pair=keys[k];long ai=m.side?pair.x:pair.y;
 long bi=cross?(m.side?pair.y:pair.x):ai;Col ca=cols[ai],cb=cols[bi];
 ulong ia=ca.word+a,ib=cb.word+b;bool av=v[ia],bv=v[ib];
 if(av!=bv)return av?1:-1;if(!av)continue;
 if(ca.type==12){ulong2 wa(x[ia],x[ia+max(ca.rows,1L)]),wb(x[ib],x[ib+max(cb.rows,1L)]);if(wide_less(wa,wb))return -1;if(wide_less(wb,wa))return 1;continue;}
 if(extrema_better(x[ia],x[ib],ca.type,false))return -1;
 if(extrema_better(x[ia],x[ib],ca.type,true))return 1;}
 return 0;
}
bool j_less(uint a,uint b,device const ulong *x,device const uchar *v,
 device const Col *cols,device const long2 *keys,constant Meta &m) {
 int c=j_cmp(a,b,false,x,v,cols,keys,m);return c<0||(c==0&&a<b);
}
kernel void j_init(device uint *index[[buffer(0)]],constant Meta &m[[buffer(1)]],uint i[[thread_position_in_grid]]){if(i<m.build)index[i]=i;}
kernel void j_merge(device const uint *a[[buffer(0)]],device uint *b[[buffer(1)]],device const ulong *x[[buffer(2)]],
 device const uchar *v[[buffer(3)]],device const Col *cols[[buffer(4)]],device const long2 *keys[[buffer(5)]],
 constant Meta &m[[buffer(6)]],constant uint &width[[buffer(7)]],uint i[[thread_position_in_grid]]){
 if(i>=m.build)return;ulong base=(ulong(i)/(2UL*width))*(2UL*width);
 uint mid=uint(min(base+width,ulong(m.build))),end=uint(min(base+2UL*width,ulong(m.build)));
 bool first=i<mid;uint start=first?mid:uint(base),stop=first?end:mid,lo=start,hi=stop;
 while(lo<hi){uint q=lo+(hi-lo)/2;if(j_less(a[q],a[i],x,v,cols,keys,m))lo=q+1;else hi=q;}
 uint own=first?uint(base):mid;b[base+(i-own)+(lo-start)]=a[i];
}
kernel void j_match(device const uint *index[[buffer(0)]],device const ulong *x[[buffer(1)]],device const uchar *v[[buffer(2)]],
 device const Col *cols[[buffer(3)]],device const long2 *keys[[buffer(4)]],constant Meta &m[[buffer(5)]],
 device ulong *range[[buffer(6)]],device ulong *count[[buffer(7)]],device atomic_int *diff[[buffer(8)]],uint i[[thread_position_in_grid]]){
 if(i>=m.probe)return;uint low=0,high=0;bool valid=true;
 for(uint k=0;k<KEYS;++k){long2 pair=keys[k];Col c=cols[m.side?pair.y:pair.x];if(!v[c.word+i])valid=false;}
 if(HOW==6){high=m.build;}else if(valid){uint lo=0,hi=m.build;
 while(lo<hi){uint q=lo+(hi-lo)/2;if(j_cmp(index[q],i,true,x,v,cols,keys,m)<0)lo=q+1;else hi=q;}low=lo;
 hi=m.build;while(lo<hi){uint q=lo+(hi-lo)/2;if(j_cmp(index[q],i,true,x,v,cols,keys,m)<=0)lo=q+1;else hi=q;}high=lo;}
 range[i]=ulong(low)|(ulong(high)<<32);ulong n=high-low;
 if(HOW==4)n=n?1:0;else if(HOW==5)n=n?0:1;else if(HOW==1||HOW==2||HOW==3)n=max(n,1UL);
 count[i]=n;if(HOW==3&&high>low){atomic_fetch_add_explicit(diff+low,1,memory_order_relaxed);atomic_fetch_sub_explicit(diff+high,1,memory_order_relaxed);}
}
// mode 0 scans probe counts, 1 scans signed coverage differences, 2 scans unmatched flags.
kernel void j_scan(device ulong *values[[buffer(0)]],device ulong *blocks[[buffer(1)]],
 device const int *diff[[buffer(2)]],device const uint *flags[[buffer(3)]],constant uint2 &args[[buffer(4)]],
 uint i[[thread_position_in_grid]],uint lane[[thread_index_in_threadgroup]],uint group[[threadgroup_position_in_grid]]){
 threadgroup ulong scan[256];ulong z=0;if(i<args.x)z=args.y==1?ulong(long(diff[i])):(args.y==2?ulong(flags[i]):values[i]);
 scan[lane]=z;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint d=1;d<256;d*=2){ulong other=lane>=d?scan[lane-d]:0;threadgroup_barrier(mem_flags::mem_threadgroup);scan[lane]+=other;threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(i<args.x)values[i]=scan[lane]-z;if(lane==255)blocks[group]=scan[lane];
}
kernel void j_offsets(device ulong *blocks[[buffer(0)]],device Meta &m[[buffer(1)]],constant uint2 &args[[buffer(2)]],uint i[[thread_position_in_grid]]){
 if(i)return;ulong sum=0;for(uint b=0;b<args.x;++b){ulong z=blocks[b];blocks[b]=sum;sum+=z;}
 if(args.y==0){m.leftTotal=sum;m.total=min(sum,m.limit);}else if(args.y==2){m.unmatched=sum;m.total=min(m.leftTotal+sum,m.limit);}
}
kernel void j_coverage(device const ulong *prefix[[buffer(0)]],device const ulong *blocks[[buffer(1)]],device const int *diff[[buffer(2)]],
 device const uint *index[[buffer(3)]],device uint *flags[[buffer(4)]],constant Meta &m[[buffer(5)]],uint i[[thread_position_in_grid]]){
 if(i<m.build)flags[index[i]]=(prefix[i]+blocks[i/256]+ulong(long(diff[i])))==0;
}
kernel void j_unmatched(device const ulong *prefix[[buffer(0)]],device const ulong *blocks[[buffer(1)]],device const uint *flags[[buffer(2)]],
 device uint *index[[buffer(3)]],constant Meta &m[[buffer(4)]],uint i[[thread_position_in_grid]]){
 if(i<m.build&&flags[i])index[prefix[i]+blocks[i/256]]=i;
}
kernel void j_gather(device const ulong *x[[buffer(0)]],device const uchar *v[[buffer(1)]],device const Col *cols[[buffer(2)]],
 device const Out *outs[[buffer(3)]],device const uint *sorted[[buffer(4)]],device const ulong *ranges[[buffer(5)]],
 device const ulong *prefix[[buffer(6)]],device const ulong *blocks[[buffer(7)]],device const uint *unmatched[[buffer(8)]],
 constant Meta &m[[buffer(9)]],device ulong *y[[buffer(10)]],device uchar *w[[buffer(11)]],constant uint &columns[[buffer(12)]],uint2 at[[thread_position_in_grid]]){
 if(at.x>=m.total||at.y>=columns)return;long left=-1,right=-1;
 if(ulong(at.x)>=m.leftTotal){right=unmatched[ulong(at.x)-m.leftTotal];}
 else{uint lo=0,hi=m.probe;while(lo<hi){uint q=lo+(hi-lo)/2;ulong start=prefix[q]+blocks[q/256];if(start<=at.x)lo=q+1;else hi=q;}
 uint probe=lo-1;ulong start=prefix[probe]+blocks[probe/256],r=ranges[probe];uint low=uint(r),high=uint(r>>32);
 long build=-1;if(HOW!=4&&HOW!=5&&high>low){uint slot=low+uint(ulong(at.x)-start);build=HOW==6?slot:sorted[slot];}
 if(m.side){right=probe;left=build;}else{left=probe;right=build;}}
 Out o=outs[at.y];long source=-1,row=-1;
 if((o.prefer&1)&&right>=0){source=o.right;row=right;}
 else if(o.left>=0&&left>=0){source=o.left;row=left;}
 else if(o.right>=0&&right>=0){source=o.right;row=right;}
 ulong dst=ulong(o.prefer>>1)*max(m.total,1UL)+at.x;bool valid=false;ulong z=0;
 if(source>=0){ulong i=cols[source].word+ulong(row);valid=v[i];if(valid)z=x[i];}y[dst]=z;w[dst]=valid;
 if(o.type==12){ulong high=0;if(valid)high=x[cols[source].word+max(cols[source].rows,1L)+ulong(row)];y[dst+max(m.total,1UL)]=high;w[dst+max(m.total,1UL)]=valid;}
}
kernel void j_pack(device const ulong *y[[buffer(0)]],device const uchar *w[[buffer(1)]],device uchar *bits[[buffer(2)]],
 constant Meta &m[[buffer(3)]],constant uint &columns[[buffer(4)]],device const Out *outs[[buffer(5)]],uint2 at[[thread_position_in_grid]]){
 ulong packed=(m.total+7)/8;if(at.x>=packed||at.y>=columns)return;uchar valid=0,value=0;
 for(uint bit=0;bit<8;++bit){ulong row=ulong(at.x)*8+bit;if(row>=m.total)break;ulong i=ulong(outs[at.y].prefer>>1)*max(m.total,1UL)+row;valid|=uchar(w[i]!=0)<<bit;value|=uchar(y[i]!=0)<<bit;}
 bits[(2UL*at.y)*packed+at.x]=valid;bits[(2UL*at.y+1)*packed+at.x]=value;
}
)MSL";
  if (hash) {
    const std::string helpers = R"MSL(
ulong j_hash_key(uint row,bool build,device const ulong *x,device const uchar *v,
 device const Col *cols,device const long2 *keys,constant Meta &m,thread bool &valid) {
 ulong hash=0x9e3779b97f4a7c15UL;valid=true;
 for(uint k=0;k<KEYS;++k){long2 pair=keys[k];
  Col c=cols[build?(m.side?pair.x:pair.y):(m.side?pair.y:pair.x)];
  ulong at=c.word+row;if(!v[at])valid=false;
  hash=key_hash(hash^canonical_key(x[at],c.type));if(c.type==12)hash=key_hash(hash^x[at+max(c.rows,1L)]);}
 return hash;
}
kernel void j_hash_build(device const ulong *x[[buffer(0)]],device const uchar *v[[buffer(1)]],
 device const Col *cols[[buffer(2)]],device const long2 *keys[[buffer(3)]],constant Meta &m[[buffer(4)]],
 device atomic_uint *owners[[buffer(5)]],device ulong *buckets[[buffer(6)]],constant uint &capacity[[buffer(7)]],uint row[[thread_position_in_grid]]){
 if(row>=m.build)return;bool valid;ulong hash=j_hash_key(row,true,x,v,cols,keys,m,valid);
 if(!valid){buckets[row]=capacity;return;}uint bucket=uint(hash)&(capacity-1);
 for(;;){uint owner=atomic_load_explicit(owners+bucket,memory_order_relaxed);
  if(owner==0xffffffffU){uint empty=0xffffffffU;
   if(atomic_compare_exchange_weak_explicit(owners+bucket,&empty,row,memory_order_relaxed,memory_order_relaxed)){buckets[row]=bucket;return;}
   owner=empty;if(owner==0xffffffffU)continue;}
  if(j_cmp(row,owner,false,x,v,cols,keys,m)==0){buckets[row]=bucket;return;}
  bucket=(bucket+1)&(capacity-1);
 }
}
kernel void j_hash_ranges(device const uint *index[[buffer(0)]],device const ulong *buckets[[buffer(1)]],
 device uint *ranges[[buffer(2)]],constant Meta &m[[buffer(3)]],constant uint &capacity[[buffer(4)]],uint row[[thread_position_in_grid]]){
 if(row>=m.build)return;uint bucket=uint(buckets[index[row]]);if(bucket==capacity)return;
 if(row==0||buckets[index[row-1]]!=bucket)ranges[2UL*bucket]=row;
 if(row+1==m.build||buckets[index[row+1]]!=bucket)ranges[2UL*bucket+1]=row+1;
}
)MSL";
    src.insert(src.find("kernel void j_match"), helpers);
    auto signature = src.find("device atomic_int *diff[[buffer(8)]],uint i[[thread_position_in_grid]])");
    src.replace(signature, std::string("device atomic_int *diff[[buffer(8)]],uint i[[thread_position_in_grid]])").size(),
      "device atomic_int *diff[[buffer(8)]],device atomic_uint *owners[[buffer(9)]],device const ulong *bucketRanges[[buffer(10)]],constant uint &capacity[[buffer(11)]],uint i[[thread_position_in_grid]])");
    auto first = src.find("else if(valid){uint lo=0,hi=m.build;");
    auto last = src.find("high=lo;}", first) + std::string("high=lo;}").size();
    src.replace(first, last-first, R"MSL(else if(valid){bool good;ulong hash=j_hash_key(i,false,x,v,cols,keys,m,good);
 uint bucket=uint(hash)&(capacity-1);for(;;){uint owner=atomic_load_explicit(owners+bucket,memory_order_relaxed);
 if(owner==0xffffffffU)break;if(j_cmp(owner,i,true,x,v,cols,keys,m)==0){ulong pair=bucketRanges[bucket];low=uint(pair);high=uint(pair>>32);break;}
 bucket=(bucket+1)&(capacity-1);}})MSL");
  }
  return src;
}
struct JSession {
  Context *context;
  DFMJoinRequest request;
  JShape shape;
  std::vector<DFMJoinInput> inputs;
  std::vector<DFMJoinKey> keys;
  std::vector<DFMJoinOutput> outputs;
  Buffers buffers;
  std::shared_ptr<Pipelines> shaders;
  id<MTLBuffer> raw, x, v, cols, key, out, meta, sorted, scratch, ranges,
      prefix, pblocks, diff, bprefix, bblocks, unmatched;
  id<MTLBuffer> radixKeys = nil, radixRanks = nil, radixHistogram = nil,
      radixTotals = nil, radixArgs = nil, hashOwners = nil, bucketRanges = nil;
  DFMStats stats{};
  bool consumed = false;
  int64_t gpu_intervals = 0;
  explicit JSession(Context &c, const DFMJoinRequest &r, JShape s)
      : context(&c), request(r), shape(std::move(s)), buffers{c} {
    dfm_context_retain(&c);
  }
  ~JSession() { dfm_context_release(context); }
};
void jbudget(Context &c, const DFMJoinRequest &r, int64_t peak,
             int64_t owned = 0) {
  auto additional = std::max<int64_t>(0, peak - owned);
  if (additional > c.headroom() ||
      (r.memory_budget_bytes >= 0 && peak > r.memory_budget_bytes))
    fail("Metal join exceeds memory budget before submission");
}
id<MTLBuffer> jmake(JSession &s, int64_t bytes) {
  if (uint64_t(std::max<int64_t>(1, bytes)) > s.context->device.maxBufferLength)
    fail("Metal join exceeds maximum buffer length before submission");
  return s.buffers.make(bytes);
}
struct JEncoder {
  JSession &s;
  Drain drain;
  id<MTLComputeCommandEncoder> encoder;
  explicit JEncoder(JSession &session)
      : s(session), drain{[session.context->queue commandBuffer]} {
    if (!drain.command)
      fail("Metal join command buffer creation failed");
    encoder = [drain.command computeCommandEncoder];
    if (!encoder)
      fail("Metal join encoder creation failed");
  }
  void buf(int i, id<MTLBuffer> b) { [encoder setBuffer:b offset:0 atIndex:i]; }
  template <class T> void val(int i, const T &v) {
    [encoder setBytes:&v length:sizeof(T) atIndex:i];
  }
  void launch(const char *name, int64_t rows, int64_t cols = 1,
              bool full = false) {
    [encoder setComputePipelineState:s.shaders->kernels.at(name)];
    if (full)
      [encoder dispatchThreadgroups:MTLSizeMake(std::max<int64_t>(
                                                    1, (rows + 255) / 256),
                                                cols, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    else
      [encoder dispatchThreads:MTLSizeMake(std::max<int64_t>(1, rows), cols, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++s.stats.memory.launches;
  }
  void wait() {
    [encoder endEncoding];
    auto start = Clock::now();
    [drain.command commit];
    drain.committed = true;
    [drain.command waitUntilCompleted];
    drain.waited = true;
    s.stats.submit_wait_ns += ns(start);
    ++s.stats.synchronizations;
    if (drain.command.status == MTLCommandBufferStatusError)
      fail("Metal join command failed: " +
           std::string(drain.command.error.localizedDescription.UTF8String));
    if (s.request.profiling &&
        drain.command.GPUEndTime >= drain.command.GPUStartTime &&
        drain.command.GPUStartTime > 0) {
      if (s.stats.gpu_ns < 0)
        s.stats.gpu_ns = 0;
      s.stats.gpu_ns += int64_t(
          (drain.command.GPUEndTime - drain.command.GPUStartTime) * 1e9);
      ++s.gpu_intervals;
    }
  }
};
void jscan(JEncoder &e, id<MTLBuffer> values, id<MTLBuffer> blocks,
           int64_t rows, uint32_t mode, id<MTLBuffer> flags) {
  e.buf(0, values);
  e.buf(1, blocks);
  e.buf(2, e.s.diff);
  e.buf(3, flags);
  std::array<uint32_t, 2> args{uint32_t(rows), mode};
  e.val(4, args);
  e.launch("j_scan", rows, 1, true);
  e.buf(0, blocks);
  e.buf(1, e.s.meta);
  args = {uint32_t(std::max<int64_t>(1, (rows + 255) / 256)), mode};
  e.val(2, args);
  e.launch("j_offsets", 1);
}
std::unique_ptr<JSession> jprepare(Context &c, const DFMJoinRequest &r,
                                  Query *query = nullptr,
                                  const DFMResidentInput *refs = nullptr,
                                  bool final = false) {
  auto shape = jvalidate(r, true, refs);
  if (query) {
    for (int64_t i = 0; i < r.input_count; ++i)
      residentInput(*query, refs, i,
          r.inputs[i].side ? r.right_rows : r.left_rows, r.inputs[i].input.dtype);
    query->check(shape.memory.peak_bytes);
  }
  jbudget(c, r, shape.memory.peak_bytes);
  auto s = std::make_unique<JSession>(c, r, shape);
  auto &q = *s;
  q.stats.memory = shape.memory;
  q.stats.memory.launches = 0;
  q.stats.gpu_ns = -1;
  if (r.input_count)
    q.inputs.assign(r.inputs, r.inputs + r.input_count);
  if (r.key_count)
    q.keys.assign(r.keys, r.keys + r.key_count);
  q.outputs.assign(r.outputs, r.outputs + r.output_count);
  q.request.inputs = q.inputs.data();
  q.request.keys = q.keys.data();
  q.request.outputs = q.outputs.data();
  auto start = Clock::now();
  bool hit = false;
  q.shaders = pipelines(c, jsource(r.key_count, r.how, shape.hash), hit);
  q.stats.pipeline_cache_hit = hit;
  q.stats.pipeline_compile_ns = hit ? 0 : ns(start);
  start = Clock::now();
  q.raw = jmake(q, shape.raw);
  q.x = jmake(q, mul(shape.words, 8));
  q.v = jmake(q, shape.words);
  q.cols = jmake(q, mul(r.input_count, 48));
  q.key = jmake(q, mul(r.key_count, 16));
  q.out = jmake(q, mul(r.output_count, 32));
  q.meta = jmake(q, 48);
  q.sorted = jmake(q, mul(shape.build, 4));
  q.scratch = jmake(q, mul(shape.build, 4));
  q.ranges = jmake(q, mul(shape.probe, 8));
  q.prefix = jmake(q, mul(shape.probe, 8));
  q.pblocks = jmake(q, mul(shape.pb, 8));
  q.diff = jmake(q, mul(add(shape.build, 1), 4));
  q.bprefix = jmake(q, mul(shape.build, 8));
  q.bblocks = jmake(q, mul(shape.bb, 8));
  q.unmatched = jmake(q, mul(shape.build, 4));
  if (shape.hash) {
    RadixShape radix(shape.build);
    auto capacity = hashCapacity(shape.build);
    q.radixKeys = jmake(q, mul(radix.cap, 8)); q.radixRanks = jmake(q, mul(radix.cap, 4));
    q.radixHistogram = jmake(q, radix.histogram); q.radixTotals = jmake(q, 64); q.radixArgs = jmake(q, 32);
    q.hashOwners = jmake(q, mul(capacity, 4)); q.bucketRanges = jmake(q, mul(capacity, 8));
    auto *args = static_cast<uint32_t *>(q.radixArgs.contents);
    args[0] = shape.build; args[1] = radix.blocks;
  }
  for (int64_t i = 0; i < r.input_count; ++i) {
    if (refs && refs[i].table)
      continue;
    auto a = r.inputs[i].input;
    auto col = shape.columns[i];
    auto packed = jpacked(col.rows, col.bit);
    auto bytes = a.dtype == DFM_BOOL ? packed : mul(col.rows, width(a.dtype));
    if (bytes)
      memcpy(static_cast<char *>(q.raw.contents) + col.values, a.values, bytes);
    if (a.has_validity && packed)
      memcpy(static_cast<char *>(q.raw.contents) + col.validity, a.validity,
             packed);
  }
  if (r.input_count)
    memcpy(q.cols.contents, shape.columns.data(), r.input_count * 48);
  if (r.key_count)
    memcpy(q.key.contents, r.keys, r.key_count * 16);
  auto *descriptors = static_cast<JOutput *>(q.out.contents);
  int64_t output_slot = 0;
  for (int64_t i = 0; i < r.output_count; ++i) {
    auto o = r.outputs[i];
    descriptors[i] = {o.dtype, o.left, o.right, 2 * output_slot + o.prefer_right};
    output_slot += o.dtype == DFM_DECIMAL128 ? 2 : 1;
  }
  *static_cast<JMeta *>(q.meta.contents) = {uint32_t(shape.probe),
                                            uint32_t(shape.build),
                                            uint32_t(r.how == 2),
                                            uint32_t(r.input_count),
                                            0,
                                            0,
                                            0,
                                            r.limit < 0 ? UINT64_MAX
                                                        : uint64_t(r.limit)};
  memset(q.diff.contents, 0, add(shape.build, 1) * 4);
  q.stats.staging_copy_ns = ns(start);
  JEncoder e(q);
  if (r.input_count) {
    e.buf(0, q.raw);
    e.buf(1, q.x);
    e.buf(2, q.v);
    e.buf(3, q.cols);
    e.buf(4, q.meta);
    e.launch("j_unpack", std::max(r.left_rows, r.right_rows), r.input_count);
  }
  if (refs)
    for (int64_t i = 0; i < r.input_count; ++i) {
      if (!refs[i].table)
        continue;
      auto &table = *static_cast<Resident *>(refs[i].table);
      auto params = residentCopy(table, refs[i].column, shape.columns[i].word, std::max<int64_t>(1, shape.columns[i].rows));
      e.buf(0, table.values);
      e.buf(1, table.validity);
      e.buf(2, q.x);
      e.buf(3, q.v);
      e.val(4, params);
      e.launch("resident_copy", table.rows);
    }
  if (r.how != 6) {
    e.buf(0, q.sorted);
    e.buf(1, q.meta);
    e.launch("j_init", shape.build);
    if (shape.hash) {
      uint32_t capacity = hashCapacity(shape.build);
      e.buf(0, q.hashOwners); e.val(1, capacity); e.launch("hash_owners_init", capacity);
      e.buf(0, q.x); e.buf(1, q.v); e.buf(2, q.cols); e.buf(3, q.key); e.buf(4, q.meta);
      e.buf(5, q.hashOwners); e.buf(6, q.radixKeys); e.val(7, capacity); e.launch("j_hash_build", shape.build);
      int bits = 0; for (auto cap = capacity; cap; cap >>= 1) ++bits;
      for (uint32_t shift = 0; shift < uint32_t(bits); shift += 4) {
        e.buf(0, q.sorted); e.buf(1, q.radixKeys); e.buf(2, q.radixRanks);
        e.buf(3, q.radixHistogram); e.buf(4, q.radixArgs); e.val(5, shift);
        e.launch("radix_rank", shape.build, 1, true);
        e.buf(0, q.radixHistogram); e.buf(1, q.radixTotals); e.buf(2, q.radixArgs);
        e.launch("radix_prefix", 16 * 256, 1, true);
        e.buf(0, q.sorted); e.buf(1, q.scratch); e.buf(2, q.radixKeys); e.buf(3, q.radixRanks);
        e.buf(4, q.radixHistogram); e.buf(5, q.radixTotals); e.buf(6, q.radixArgs); e.val(7, shift);
        e.launch("radix_scatter", shape.build); std::swap(q.sorted, q.scratch);
      }
      e.buf(0, q.sorted); e.buf(1, q.radixKeys); e.buf(2, q.bucketRanges); e.buf(3, q.meta); e.val(4, capacity);
      e.launch("j_hash_ranges", shape.build);
    } else
    for (uint64_t width = 1; width < uint64_t(shape.build); width *= 2) {
      e.buf(0, q.sorted);
      e.buf(1, q.scratch);
      e.buf(2, q.x);
      e.buf(3, q.v);
      e.buf(4, q.cols);
      e.buf(5, q.key);
      e.buf(6, q.meta);
      e.val(7, uint32_t(width));
      e.launch("j_merge", shape.build);
      std::swap(q.sorted, q.scratch);
    }
  }
  e.buf(0, q.sorted);
  e.buf(1, q.x);
  e.buf(2, q.v);
  e.buf(3, q.cols);
  e.buf(4, q.key);
  e.buf(5, q.meta);
  e.buf(6, q.ranges);
  e.buf(7, q.prefix);
  e.buf(8, q.diff);
  if (shape.hash) {
    e.buf(9, q.hashOwners); e.buf(10, q.bucketRanges); e.val(11, uint32_t(hashCapacity(shape.build)));
  }
  e.launch("j_match", shape.probe);
  jscan(e, q.prefix, q.pblocks, shape.probe, 0, q.scratch);
  if (r.how == 3) {
    jscan(e, q.bprefix, q.bblocks, shape.build, 1, q.scratch);
    e.buf(0, q.bprefix);
    e.buf(1, q.bblocks);
    e.buf(2, q.diff);
    e.buf(3, q.sorted);
    e.buf(4, q.scratch);
    e.buf(5, q.meta);
    e.launch("j_coverage", shape.build);
    jscan(e, q.bprefix, q.bblocks, shape.build, 2, q.scratch);
    e.buf(0, q.bprefix);
    e.buf(1, q.bblocks);
    e.buf(2, q.scratch);
    e.buf(3, q.unmatched);
    e.buf(4, q.meta);
    e.launch("j_unmatched", shape.build);
  }
  e.wait();
  auto total = static_cast<JMeta *>(q.meta.contents)->total;
  if (total > INT32_MAX)
    fail("Metal join output exceeds supported row count");
  q.stats.output_rows = int64_t(total);
  q.stats.download_bytes = sizeof(uint64_t);
  // Check the complete peak before the caller allocates output storage.
  auto rows = q.stats.output_rows, packed = jpacked(rows);
  int64_t result = 0;
  if (!query || final)
    for (auto o : q.outputs)
      result = add(
          result,
          add(packed, o.dtype == DFM_BOOL ? packed : mul(rows, width(o.dtype))));
  auto words = mul(mul(std::max<int64_t>(1, rows), jOutputSlots(r)), 8);
  auto valid = mul(std::max<int64_t>(1, rows), jOutputSlots(r)),
       bits = query && !final ? 1 : mul(mul(packed, r.output_count), 2);
  auto shared =
      add(shape.memory.shared_bytes,
          add(add(std::max<int64_t>(1, words), std::max<int64_t>(1, valid)),
              std::max<int64_t>(1, bits)));
  auto peak = add(shared, result);
  if (query)
    query->check(peak);
  jbudget(c, r, peak, q.buffers.requested);
  for (auto n : {words, valid, bits})
    if (uint64_t(std::max<int64_t>(1, n)) > c.device.maxBufferLength)
      fail("Metal join exceeds maximum buffer length before submission");
  q.stats.memory.shared_bytes = shared;
  q.stats.memory.result_bytes = result;
  q.stats.memory.peak_bytes = peak;
  q.stats.memory.packed_bytes = bits;
  q.stats.memory.capacity = std::max<int64_t>(1, rows);
  return s;
}
void jfinish(JSession &q, const DFMJoinOutput *outputs, int64_t count,
             Query *query = nullptr, Resident **resident = nullptr,
             bool final = false) {
  if (q.consumed)
    fail("Metal join session already consumed");
  q.consumed = true;
  auto &r = q.request;
  auto rows = q.stats.output_rows, packed = jpacked(rows);
  if (!outputs || count != r.output_count)
    fail("Invalid Metal join finish outputs");
  int64_t result = 0;
  for (int64_t i = 0; i < count; ++i) {
    auto a = outputs[i], b = q.outputs[i];
    if (a.dtype != b.dtype || a.left != b.left || a.right != b.right ||
        a.prefer_right != b.prefer_right ||
        (!query && rows && (!a.values || !a.validity)))
      fail("Metal join finish output metadata mismatch");
    if (!query || final)
      result = add(
          result,
          add(packed, a.dtype == DFM_BOOL ? packed : mul(rows, width(a.dtype))));
  }
  auto words = mul(mul(std::max<int64_t>(1, rows), jOutputSlots(r)), 8),
       valid = mul(std::max<int64_t>(1, rows), jOutputSlots(r)),
       bits = query && !final ? 1 : mul(mul(packed, count), 2);
  auto shared =
      add(q.shape.memory.shared_bytes,
          add(add(std::max<int64_t>(1, words), std::max<int64_t>(1, valid)),
              std::max<int64_t>(1, bits)));
  auto peak = add(shared, result);
  if (query)
    query->check(peak);
  jbudget(*q.context, r, peak, q.buffers.requested);
  // Validate every output allocation before requesting any of them.
  for (auto n : {words, valid, bits})
    if (uint64_t(std::max<int64_t>(1, n)) > q.context->device.maxBufferLength)
      fail("Metal join exceeds maximum buffer length before submission");
  auto y = jmake(q, words), w = jmake(q, valid), bitmap = jmake(q, bits);
  JEncoder e(q);
  e.buf(0, q.x);
  e.buf(1, q.v);
  e.buf(2, q.cols);
  e.buf(3, q.out);
  e.buf(4, q.sorted);
  e.buf(5, q.ranges);
  e.buf(6, q.prefix);
  e.buf(7, q.pblocks);
  e.buf(8, q.unmatched);
  e.buf(9, q.meta);
  e.buf(10, y);
  e.buf(11, w);
  e.val(12, uint32_t(count));
  e.launch("j_gather", rows, count);
  if (!query || final) {
    e.buf(0, y);
    e.buf(1, w);
    e.buf(2, bitmap);
    e.buf(3, q.meta);
    e.val(4, uint32_t(count));
    e.buf(5, q.out);
    e.launch("j_pack", packed, count);
  }
  e.wait();
  if (query) {
    auto table = std::make_unique<Resident>(*query);
    table->rows = rows;
    table->cap = std::max<int64_t>(1, rows);
    table->packed = packed;
    table->final = final;
    table->values = y;
    table->validity = w;
    q.buffers.transfer(y, table->buffers);
    q.buffers.transfer(w, table->buffers);
    if (final) {
      table->bitmap = bitmap;
      q.buffers.transfer(bitmap, table->buffers);
    }
    for (int64_t i = 0; i < count; ++i)
      table->columns.push_back({static_cast<JOutput *>(q.out.contents)[i].prefer >> 1, outputs[i].dtype, -1, 0});
    q.stats.memory.shared_bytes = shared;
    q.stats.memory.result_bytes = result;
    q.stats.memory.peak_bytes = peak;
    q.stats.memory.packed_bytes = bits;
    q.stats.memory.capacity = table->cap;
    q.stats.result_copy_ns = 0;
    if (q.gpu_intervals != q.stats.synchronizations)
      q.stats.gpu_ns = -1;
    table->stats = q.stats;
    table->seal();
    *resident = table.release();
    return;
  }
  auto start = Clock::now();
  for (int64_t i = 0; i < count; ++i) {
    auto o = outputs[i];
    auto *base = static_cast<uint8_t *>(bitmap.contents) + 2 * i * packed;
    if (packed)
      memcpy(o.validity, base, packed);
    if (o.dtype == DFM_BOOL) {
      if (packed)
        memcpy(o.values, base + packed, packed);
    } else {
      auto *source =
          static_cast<uint64_t *>(y.contents) + (static_cast<JOutput *>(q.out.contents)[i].prefer >> 1) * std::max<int64_t>(1, rows);
      auto widthBytes = width(o.dtype);
      for (int64_t row = 0; row < rows; ++row)
        memcpy(static_cast<char *>(o.values) + row * widthBytes, source + row,
               std::min<int64_t>(8, widthBytes));
      if (o.dtype == DFM_DECIMAL128)
        for (int64_t row = 0; row < rows; ++row)
          memcpy(static_cast<char *>(o.values) + row * 16 + 8, source + std::max<int64_t>(1, rows) + row, 8);
    }
  }
  q.stats.result_copy_ns = ns(start);
  q.stats.download_bytes = add(q.stats.download_bytes, result);
  if (q.gpu_intervals != q.stats.synchronizations)
    q.stats.gpu_ns = -1;
  q.stats.memory.shared_bytes = shared;
  q.stats.memory.result_bytes = result;
  q.stats.memory.peak_bytes = peak;
  q.stats.memory.packed_bytes = bits;
  q.stats.memory.capacity = std::max<int64_t>(1, rows);
}
} // namespace
extern "C" {
int32_t dfm_join_estimate(const DFMJoinRequest *r, DFMMemory *m, char **err) {
  if (err)
    *err = nullptr;
  try {
    if (!r || !m)
      fail("Missing Metal join estimate request");
    *m = jvalidate(*r).memory;
    return 0;
  } catch (const std::exception &e) {
    error(err, e);
    return 1;
  }
}
int64_t dfm_join_plan_features(const DFMJoinRequest *r, char **err) {
  if (err) *err = nullptr;
  try {
    if (!r) fail("Missing Metal join feature request");
    return jvalidate(*r).hash ? 1 : 0;
  } catch (const std::exception &e) { error(err, e); return -1; }
}
int32_t dfm_join_prepare(void *context, const DFMJoinRequest *r, void **session,
                         DFMStats *stats, char **err) {
  if (err)
    *err = nullptr;
  if (session)
    *session = nullptr;
  @autoreleasepool {
    try {
      if (!context || !r || !session || !stats)
        fail("Missing Metal join prepare request");
      auto &c = *static_cast<Context *>(context);
      std::lock_guard<std::mutex> lock(c.mutex);
      auto q = jprepare(c, *r);
      *stats = q->stats;
      *session = q.release();
      return 0;
    } catch (const std::exception &e) {
      error(err, e);
      return 1;
    }
  }
}
int32_t dfm_join_finish(void *session, const DFMJoinOutput *outputs,
                        int64_t count, DFMStats *stats, char **err) {
  if (err)
    *err = nullptr;
  @autoreleasepool {
    try {
      if (!session || !stats)
        fail("Missing Metal join finish request");
      auto &q = *static_cast<JSession *>(session);
      std::lock_guard<std::mutex> lock(q.context->mutex);
      jfinish(q, outputs, count);
      *stats = q.stats;
      return 0;
    } catch (const std::exception &e) {
      error(err, e);
      return 1;
    }
  }
}
void dfm_join_release(void *session) {
  delete static_cast<JSession *>(session);
}
int32_t dfm_resident_join(void *p, const DFMJoinRequest *r,
    const DFMResidentInput *refs, int64_t count, int64_t final,
    void **table, DFMStats *stats, char **err) {
  if (err)
    *err = nullptr;
  if (table)
    *table = nullptr;
  @autoreleasepool {
    try {
      if (!p || !r || !table || !stats || count != r->input_count ||
          (count && !refs) || (final != 0 && final != 1))
        fail("Invalid Metal resident join request");
      auto &query = *static_cast<Query *>(p);
      std::lock_guard<std::mutex> lock(query.context.mutex);
      auto q = jprepare(query.context, *r, &query, refs, final);
      Resident *result = nullptr;
      jfinish(*q, q->outputs.data(), r->output_count, &query, &result, final);
      *stats = q->stats;
      *table = result;
      return 0;
    } catch (const std::exception &e) {
      error(err, e);
      return 1;
    }
  }
}
}
