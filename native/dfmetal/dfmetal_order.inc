// Shared stable radix primitives for resident rows and joins. Each digit pass
// preserves input-index order, so equal keys keep their original row order.
struct RadixShape {
  int64_t cap, blocks, histogram;
  explicit RadixShape(int64_t rows)
      : cap(std::max<int64_t>(1, rows)), blocks((cap + 255) / 256),
        histogram(mul(mul(blocks, 16), 4)) {}
  int64_t bytes() const {
    return add(add(mul(cap, 8), mul(cap, 4)), add(histogram, 96));
  }
};
int64_t hashCapacity(int64_t rows) {
  int64_t capacity = 1;
  while (capacity < mul(std::max<int64_t>(1, rows), 2))
    capacity *= 2;
  return capacity;
}
bool hashCandidate(int64_t rows) {
  // Keep the optional table bounded independently of input buffer limits.
  return rows >= 4096 && rows <= 8388608;
}
std::string radixSource() {
  return R"MSL(
ulong key_hash(ulong value){
 value^=value>>30;value*=0xbf58476d1ce4e5b9UL;
 value^=value>>27;value*=0x94d049bb133111ebUL;return value^(value>>31);
}
ulong canonical_key(ulong word,long type){
 if(type==11){ulong a=word&0x7fffffffffffffffUL;
  if(a==0)return 0;if(a>0x7ff0000000000000UL)return 0x7ff8000000000000UL;return word;}
 if(type==1){uint x=uint(word),a=x&0x7fffffffU;
  if(a==0)return 0;if(a>0x7f800000U)return 0x7fc00000UL;return x;}
 return word;
}
kernel void hash_owners_init(device atomic_uint *owners[[buffer(0)]],
 constant uint &capacity[[buffer(1)]],uint row[[thread_position_in_grid]]) {
 if(row<capacity)atomic_store_explicit(owners+row,0xffffffffU,memory_order_relaxed);
}
struct RadixArgs { uint rows, blocks, shift, unused; };
kernel void radix_rank(device const uint *index[[buffer(0)]],
 device const ulong *keys[[buffer(1)]],device uint *rank[[buffer(2)]],
 device uint *histogram[[buffer(3)]],constant RadixArgs &p[[buffer(4)]],
 constant uint &shift[[buffer(5)]],
 uint row[[thread_position_in_grid]],uint lane[[thread_index_in_threadgroup]],
 uint group[[threadgroup_position_in_grid]]) {
 if(group>=(p.rows+255)/256)return;
 threadgroup ulong4 scan[256];uint digit=16;
 if(row<p.rows)digit=uint((keys[index[row]]>>shift)&15UL);
 ulong4 value=ulong4(0);if(digit<16)value[digit/4]=1UL<<(16*(digit%4));
 scan[lane]=value;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint d=1;d<256;d*=2){ulong4 previous=lane>=d?scan[lane-d]:ulong4(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);scan[lane]+=previous;
  threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(row<p.rows)rank[row]=uint((scan[lane][digit/4]>>(16*(digit%4)))&65535UL)-1;
 if(lane<16)histogram[ulong(lane)*p.blocks+group]=
  uint((scan[255][lane/4]>>(16*(lane%4)))&65535UL);
}
kernel void radix_prefix(device uint *histogram[[buffer(0)]],
 device uint *totals[[buffer(1)]],constant RadixArgs &p[[buffer(2)]],
 uint lane[[thread_index_in_threadgroup]],uint digit[[threadgroup_position_in_grid]]) {
 if(digit>=16)return;threadgroup uint scan[256];threadgroup uint carry;
 if(lane==0)carry=0;threadgroup_barrier(mem_flags::mem_threadgroup);
 uint blocks=(p.rows+255)/256;
 for(uint first=0;first<blocks;first+=256){uint at=first+lane;
  uint value=at<blocks?histogram[ulong(digit)*p.blocks+at]:0;
  scan[lane]=value;threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint d=1;d<256;d*=2){uint previous=lane>=d?scan[lane-d]:0;
   threadgroup_barrier(mem_flags::mem_threadgroup);scan[lane]+=previous;
   threadgroup_barrier(mem_flags::mem_threadgroup);}
  uint base=carry;if(at<blocks)histogram[ulong(digit)*p.blocks+at]=base+scan[lane]-value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==255)carry=base+scan[lane];threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(lane==0)totals[digit]=carry;
}
kernel void radix_scatter(device const uint *index[[buffer(0)]],
 device uint *target[[buffer(1)]],device const ulong *keys[[buffer(2)]],
 device const uint *rank[[buffer(3)]],device const uint *histogram[[buffer(4)]],
 device const uint *totals[[buffer(5)]],constant RadixArgs &p[[buffer(6)]],
 constant uint &shift[[buffer(7)]],
 uint row[[thread_position_in_grid]]) {
 if(row>=p.rows)return;uint digit=uint((keys[index[row]]>>shift)&15UL),base=0;
 for(uint d=0;d<digit;++d)base+=totals[d];
 target[base+histogram[ulong(digit)*p.blocks+row/256]+rank[row]]=index[row];
}
// Row counts can be GPU-produced after a filter/group. Indirect dispatch keeps
// radix work proportional to that live count while allocation remains bounded.
kernel void radix_args(constant uint4 &meta[[buffer(0)]],
 device uint4 *args[[buffer(1)]],constant uint &blocks[[buffer(2)]],
 uint i[[thread_position_in_grid]]) {
 if(i==0){args[0]=uint4(meta.x,blocks,0,0);
 args[1]=uint4(max(1U,(meta.x+255)/256),1,1,0);}
}
)MSL";
}
