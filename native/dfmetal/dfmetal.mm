#include "dfmetal.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef DFM_BUILD_ID
#define DFM_BUILD_ID "development"
#endif
static_assert(sizeof(DFMInput) == 40 && sizeof(DFMStep) == 48);
static_assert(sizeof(DFMOutput) == 48 && sizeof(DFMRequest) == 160);
static_assert(sizeof(DFMMemory) == 64 && sizeof(DFMStats) == 136);
static_assert(sizeof(DFMDeviceInfo) == 48);
namespace {
using Clock = std::chrono::steady_clock;
int64_t ns(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                              start)
      .count();
}
void fail(const std::string &s) { throw std::runtime_error(s); }
void error(char **p, const std::exception &e) {
  if (p)
    *p = strdup(e.what());
}
int64_t add(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || a > INT64_MAX - b)
    fail("Metal allocation size overflow");
  return a + b;
}
int64_t mul(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || (b && a > INT64_MAX / b))
    fail("Metal allocation size overflow");
  return a * b;
}
int64_t width(int64_t t) {
  if (t == DFM_FLOAT32 || t == DFM_INT32)
    return 4;
  if (t == DFM_INT64)
    return 8;
  if (t == DFM_BOOL)
    return 1;
  fail("Metal unsupported dtype");
  return 0;
}
struct Shape {
  int64_t cap, packed, blocks, bitmap, matrix, validity, literal, outputBits,
      partial;
  bool filter;
};
Shape validate(const DFMRequest &r, bool storage = false) {
  if (r.abi_version != DFM_ABI_VERSION)
    fail("Metal ABI version mismatch");
  if (r.rows < 0 || r.rows > INT32_MAX || r.slots < 1 || r.slots > 64 ||
      r.input_count < 1 || r.input_count > r.slots || r.step_count < 0 ||
      r.step_count > 64 || r.output_count < 1 || r.output_count > 64 ||
      r.literal_count < 0 || r.literal_count > 4096 ||
      r.code_words != mul(r.literal_count, 4) || r.gather_count < 0 ||
      r.gather_count > 4096 || r.limit < -1 || r.memory_budget_bytes < -1 ||
      (r.reductions != 0 && r.reductions != 1))
    fail("Invalid Metal request shape");
  if (r.dtype != DFM_FLOAT32 && r.dtype != DFM_INT32 && r.dtype != DFM_INT64)
    fail("Metal requires native Float32, Int32 or Int64 storage");
  if (!r.inputs || !r.outputs || (r.step_count && !r.steps) ||
      (r.literal_count && (!r.code || !r.literals)) ||
      (r.gather_count && !r.gathers))
    fail("Missing Metal request descriptors");
  Shape s{};
  s.cap = std::max<int64_t>(1, r.rows);
  s.packed = (s.cap + 7) / 8;
  s.blocks = (s.cap + 255) / 256;
  for (int64_t i = 0; i < r.input_count; ++i) {
    const auto &v = r.inputs[i];
    width(v.dtype);
    if (v.dtype != DFM_BOOL && v.dtype != r.dtype)
      fail("Mixed Metal input dtypes");
    if (v.bit_offset < 0 || v.bit_offset > 7 ||
        (v.has_validity != 0 && v.has_validity != 1))
      fail("Invalid input bitmap descriptor");
    if (storage && r.rows && (!v.values || (v.has_validity && !v.validity)))
      fail("Missing Metal input data");
    int64_t bytes = (r.rows + v.bit_offset + 7) / 8;
    if (v.dtype == DFM_BOOL)
      s.bitmap = add(s.bitmap, bytes);
    if (v.has_validity)
      s.bitmap = add(s.bitmap, bytes);
  }
  for (int64_t i = 0; i < r.step_count; ++i) {
    const auto &v = r.steps[i];
    if (v.start < 0 || v.nodes < 1 || v.nodes > 64 ||
        add(v.start, v.nodes) > r.literal_count || v.slot < r.input_count ||
        v.slot >= r.slots || v.gather_start < 0 || v.gather_count < 0 ||
        add(v.gather_start, v.gather_count) > r.gather_count ||
        (v.filter != 0 && v.filter != 1))
      fail("Invalid Metal expression step");
    s.filter |= v.filter;
    for (int64_t j = 0; j < v.nodes; ++j) {
      const auto *c = r.code + 4 * (v.start + j);
      bool binary = c[0] == DFM_ADD || c[0] == DFM_SUB || c[0] == DFM_MUL ||
                    (c[0] >= DFM_GT && c[0] <= DFM_EQ) ||
                    (c[0] >= DFM_LT && c[0] <= DFM_NE) ||
                    (c[0] >= DFM_AND && c[0] <= DFM_FILL_NULL);
      bool unary =
          c[0] == DFM_NEG || (c[0] >= DFM_NOT && c[0] <= DFM_IS_NOT_NULL);
      bool leaf = c[0] == DFM_COLUMN || c[0] == DFM_LITERAL_INT ||
                  c[0] == DFM_LITERAL_FLOAT || c[0] == DFM_LITERAL_BOOL ||
                  c[0] == DFM_LITERAL_NULL;
      if ((!binary && !unary && !leaf) ||
          ((binary || unary) && (c[1] < 0 || c[1] >= j)) ||
          (binary && (c[2] < 0 || c[2] >= j)) ||
          (c[0] == DFM_COLUMN && (c[3] < 0 || c[3] >= v.slot)))
        fail("Invalid Metal expression bytecode");
      if (r.dtype != DFM_FLOAT32 && c[0] == DFM_LITERAL_FLOAT)
        fail("Floating literal requires native Float32 storage");
    }
  }
  for (int64_t i = 0; i < r.gather_count; ++i)
    if (r.gathers[i] < 0 || r.gathers[i] >= r.slots)
      fail("Invalid Metal gather slot");
  for (int64_t i = 0; i < r.output_count; ++i) {
    const auto &v = r.outputs[i];
    width(v.dtype);
    if (v.slot < 0 || v.slot >= r.slots || v.min_count < 0)
      fail("Invalid Metal output descriptor");
    if (!r.reductions) {
      if (v.reduction != -1 || (v.dtype != r.dtype && v.dtype != DFM_BOOL))
        fail("Invalid Metal row output");
    } else {
      if (v.reduction != DFM_COUNT && v.reduction != DFM_LEN &&
          !(v.reduction == DFM_SUM && r.dtype == DFM_INT32))
        fail("Metal reduction requires unsupported accumulator precision");
      if (v.dtype != (v.reduction == DFM_SUM ? DFM_INT32 : DFM_INT64))
        fail("Invalid Metal reduction dtype");
    }
    if (storage && (r.rows || r.reductions) && (!v.values || !v.validity))
      fail("Missing Metal output storage");
  }
  s.matrix = mul(mul(s.cap, r.slots), width(r.dtype));
  s.validity = mul(s.cap, r.slots);
  s.literal = std::max<int64_t>(1, mul(r.literal_count, width(r.dtype)));
  s.outputBits = mul(mul(s.packed, r.output_count), 2);
  s.partial =
      r.reductions
          ? mul(mul(std::min<int64_t>(s.blocks, 1024), r.output_count), 16)
          : 1;
  return s;
}
DFMMemory estimate(const DFMRequest &r, const Shape &s) {
  DFMMemory m{};
  m.capacity = s.cap;
  m.packed_bytes = s.packed;
  m.blocks = s.blocks;
  auto account = [&](int64_t bytes) {
    m.shared_bytes = add(m.shared_bytes, std::max<int64_t>(1, bytes));
  };
  account(s.matrix);
  account(s.validity);
  if (s.filter) {
    account(s.matrix);
    account(s.validity);
    account(mul(s.cap, 4));
    account(mul(s.blocks, 4));
  }
  account(s.bitmap);
  account(s.literal);
  account(s.outputBits);
  account(s.partial);
  account(mul(r.output_count, 32));
  account(16);
  account(4);
  if (r.reductions)
    account(mul(r.output_count, 16));
  for (int64_t i = 0; i < r.input_count; ++i) {
    if (r.inputs[i].dtype != DFM_BOOL)
      m.upload_bytes = add(m.upload_bytes, mul(r.rows, width(r.dtype)));
  }
  m.upload_bytes = add(m.upload_bytes, s.bitmap);
  m.upload_bytes = add(m.upload_bytes, mul(r.literal_count, width(r.dtype)));
  for (int64_t i = 0; i < r.output_count; ++i) {
    auto t = r.outputs[i].dtype;
    m.result_bytes =
        add(m.result_bytes,
            r.reductions ? width(t) + 1
                         : add(t == DFM_BOOL ? s.packed : mul(r.rows, width(t)),
                               s.packed));
  }
  m.peak_bytes = add(m.shared_bytes, m.result_bytes);
  m.launches = r.input_count + r.step_count + (r.reductions ? 2 : 1);
  for (int64_t i = 0; i < r.step_count; ++i)
    if (r.steps[i].filter)
      m.launches += 2 + r.steps[i].gather_count;
  return m;
}
struct Pipelines {
  id<MTLLibrary> library;
  std::unordered_map<std::string, id<MTLComputePipelineState>> kernels;
  uint64_t age = 0;
};
struct Context {
  std::atomic<int64_t> refs{1};
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  std::string name;
  std::mutex mutex;
  uint64_t age = 0;
  std::unordered_map<std::string, std::shared_ptr<Pipelines>> cache;
};
std::mutex defaultMutex;
Context *defaultContext = nullptr;
std::string source(const DFMRequest &r) {
  std::string text = "#include <metal_stdlib>\nusing namespace metal;\n#pragma "
                     "clang fp contract(off)\n";
  text += std::string("typedef ") +
          (r.dtype == DFM_FLOAT32 ? "float"
           : r.dtype == DFM_INT32 ? "int"
                                  : "long") +
          " T;\n";
  text += R"MSL(
struct Out { long slot; long kind; long minimum; long type; };
struct Partial { long total; long count; };
// Unsigned magnitudes avoid signed overflow, including the minimum Int64.
bool checked_add(T a,T b,thread T &z) {
)MSL";
  if (r.dtype == DFM_FLOAT32) {
    // Metal permits FTZ even in safe mode. Reject value-dependent precision
    // requirements instead of silently flushing or emulating arithmetic.
    text += R"MSL(
uint ab=as_type<uint>(a)&0x7fffffffu,bb=as_type<uint>(b)&0x7fffffffu;
if((ab&&ab<0x00800000u)||(bb&&bb<0x00800000u))return false;
z=a+b;uint zb=as_type<uint>(z)&0x7fffffffu;
return !(zb&&zb<0x00800000u)&&!(zb==0&&a!=-b);
}
bool checked_sub(T a,T b,thread T &z){
uint ab=as_type<uint>(a)&0x7fffffffu,bb=as_type<uint>(b)&0x7fffffffu;
if((ab&&ab<0x00800000u)||(bb&&bb<0x00800000u))return false;
z=a-b;uint zb=as_type<uint>(z)&0x7fffffffu;
return !(zb&&zb<0x00800000u)&&!(zb==0&&a!=b);
}
bool checked_mul(T a,T b,thread T &z){
uint ab=as_type<uint>(a)&0x7fffffffu,bb=as_type<uint>(b)&0x7fffffffu;
if((ab&&ab<0x00800000u)||(bb&&bb<0x00800000u))return false;
z=a*b;uint zb=as_type<uint>(z)&0x7fffffffu;
return !(zb&&zb<0x00800000u)&&!(zb==0&&ab&&bb);
}
bool checked_neg(T a,thread T &z){z=as_type<float>(as_type<uint>(a)^0x80000000u);return true;}
bool native_operand(T a){uint b=as_type<uint>(a)&0x7fffffffu;return b==0||b>=0x00800000u;}
)MSL";
  } else {
    std::string lo = r.dtype == DFM_INT32 ? "(-2147483647L-1L)"
                                          : "(-9223372036854775807L-1L)";
    std::string hi =
        r.dtype == DFM_INT32 ? "2147483647L" : "9223372036854775807L";
    text += "const long lo=" + lo + ",hi=" + hi +
            ";long x=a,y=b;if((y>0&&x>hi-y)||(y<0&&x<lo-y))return "
            "false;z=T(x+y);return true;}\n";
    text += "bool checked_sub(T a,T b,thread T &z){const long lo=" + lo +
            ",hi=" + hi +
            ";long x=a,y=b;if((y<0&&x>hi+y)||(y>0&&x<lo+y))return "
            "false;z=T(x-y);return true;}\n";
    text += "bool checked_mul(T a,T b,thread T &z){bool neg=(a<0)!=(b<0);ulong "
            "x=a<0?0UL-ulong(a):ulong(a),y=b<0?0UL-ulong(b):ulong(b);ulong "
            "limit=neg?ulong(" +
            hi + ")+1UL:ulong(" + hi +
            ");if(y&&x>limit/y)return false;ulong "
            "u=x*y;z=T(neg?0UL-u:u);return true;}\n";
    text += "bool checked_neg(T a,thread T &z){if(long(a)==" + lo +
            ")return false;z=-a;return true;}\n";
  }
  text += R"MSL(
kernel void unpack(device T *x[[buffer(0)]],device uchar *v[[buffer(1)]],
 device const uchar *bits[[buffer(2)]],constant ulong4 &p[[buffer(3)]],
 constant uint4 &meta[[buffer(4)]],uint row[[thread_position_in_grid]]) {
 if(row>=meta.x)return;ulong at=p.x*meta.z+row;ulong bit=row+p.y;
 if(p.z!=0)x[at]=T((bits[p.z-1+bit/8]>>(bit%8))&1);
 v[at]=p.w==ulong(-1)?1:uchar((bits[p.w+bit/8]>>(bit%8))&1);
}
kernel void rank_rows(device const T *x[[buffer(0)]],device const uchar *v[[buffer(1)]],
 device uint *rank[[buffer(2)]],device uint *blocks[[buffer(3)]],
 constant uint4 &meta[[buffer(4)]],constant ulong &slot[[buffer(5)]],
 uint row[[thread_position_in_grid]],uint lane[[thread_index_in_threadgroup]],
 uint group[[threadgroup_position_in_grid]]) {
 threadgroup uint scan[256];uint yes=0;
 if(row<meta.x)yes=v[slot*meta.z+row]&&x[slot*meta.z+row]!=T(0);
 scan[lane]=yes;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint offset=1;offset<256;offset*=2){uint a=lane>=offset?scan[lane-offset]:0;
 threadgroup_barrier(mem_flags::mem_threadgroup);scan[lane]+=a;threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(row<meta.x)rank[row]=scan[lane]-yes;
 if(lane==255)blocks[group]=scan[lane];
}
kernel void block_offsets(device uint *blocks[[buffer(0)]],device uint4 &meta[[buffer(1)]]) {
 uint total=0;for(uint i=0;i<(meta.x+255)/256;++i){uint count=blocks[i];blocks[i]=total;total+=count;}
 meta.y=meta.x;meta.x=total;
}
kernel void gather_rows(device const T *x[[buffer(0)]],device const uchar *v[[buffer(1)]],
 device T *y[[buffer(2)]],device uchar *w[[buffer(3)]],device const uint *rank[[buffer(4)]],
 device const uint *blocks[[buffer(5)]],constant uint4 &meta[[buffer(6)]],
 constant ulong2 &p[[buffer(7)]],uint row[[thread_position_in_grid]]) {
 if(row>=meta.y||!v[p.x*meta.z+row]||x[p.x*meta.z+row]==T(0))return;
 ulong target=rank[row]+blocks[row/256];y[p.y*meta.z+target]=x[p.y*meta.z+row];w[p.y*meta.z+target]=v[p.y*meta.z+row];
}
kernel void pack_rows(device const T *x[[buffer(0)]],device const uchar *v[[buffer(1)]],
 device uchar *bits[[buffer(2)]],device const Out *out[[buffer(3)]],
 constant uint4 &meta[[buffer(4)]],constant ulong4 &p[[buffer(5)]],uint2 at[[thread_position_in_grid]]) {
 if(at.x>=p.x||at.y>=p.y)return;uint n=min(ulong(meta.x),p.z);uchar valid=0,value=0;
 for(uint b=0;b<8;++b){uint row=at.x*8+b;if(row<n){ulong idx=out[at.y].slot*meta.z+row;
 valid|=uchar(v[idx]!=0)<<b;value|=uchar(x[idx]!=T(0))<<b;}}
 bits[(2*at.y)*p.x+at.x]=valid;bits[(2*at.y+1)*p.x+at.x]=value;
}
kernel void reduce_rows(device const T *x[[buffer(0)]],device const uchar *v[[buffer(1)]],
 device Partial *partials[[buffer(2)]],device const Out *out[[buffer(3)]],
 constant uint4 &meta[[buffer(4)]],constant ulong &groups[[buffer(5)]],
 uint2 group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_threadgroup]]) {
 threadgroup long sums[256];threadgroup long counts[256];long sum=0,count=0;
 for(ulong row=group.x*256+lane;row<meta.x;row+=groups*256){ulong idx=out[group.y].slot*meta.z+row;
 if(v[idx]){++count;if(out[group.y].kind==10)sum+=long(x[idx]);}}
 sums[lane]=sum;counts[lane]=count;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint stride=128;stride;stride/=2){if(lane<stride){sums[lane]+=sums[lane+stride];counts[lane]+=counts[lane+stride];}
 threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(lane==0)partials[group.y*groups+group.x]={sums[0],counts[0]};
}
kernel void finish_reduce(device Partial *partials[[buffer(0)]],device Partial *result[[buffer(1)]],
 device const Out *out[[buffer(2)]],constant uint4 &meta[[buffer(3)]],constant ulong &groups[[buffer(4)]],
 uint column[[thread_position_in_grid]]) {
 long sum=0,count=0;for(ulong i=0;i<groups;++i){sum+=partials[column*groups+i].total;count+=partials[column*groups+i].count;}
 if(out[column].kind==11){sum=count;count=1;}if(out[column].kind==90){sum=meta.x;count=1;}
 result[column]={sum,count};
}
)MSL";
  for (int64_t k = 0; k < r.step_count; ++k) {
    auto &s = r.steps[k];
    text += "kernel void project" + std::to_string(k) +
            "(device T *x[[buffer(0)]],device uchar *v[[buffer(1)]],device "
            "const T *lit[[buffer(2)]],constant uint4 "
            "&meta[[buffer(3)]],device atomic_uint *error[[buffer(4)]],uint "
            "row[[thread_position_in_grid]]){if(row>=meta.x)return;\n";
    for (int64_t j = 0; j < s.nodes; ++j) {
      auto *c = r.code + 4 * (s.start + j);
      auto op = c[0];
      auto id = std::to_string(j);
      std::string a = "a" + std::to_string(c[1]),
                  b = "a" + std::to_string(c[2]);
      std::string av = "v" + std::to_string(c[1]),
                  bv = "v" + std::to_string(c[2]);
      text += "T a" + id + "=T(0);bool v" + id + "=true;";
      std::string z = "a" + id, valid = "v" + id;
      if (op == DFM_COLUMN) {
        auto idx = std::to_string(c[3]) + "UL*meta.z+row";
        text += z + "=x[" + idx + "];" + valid + "=v[" + idx + "]!=0;";
      } else if (op == DFM_LITERAL_NULL)
        text += valid + "=false;";
      else if (op == DFM_LITERAL_INT || op == DFM_LITERAL_FLOAT ||
               op == DFM_LITERAL_BOOL)
        text += z + "=lit[" + std::to_string(s.start + j) + "];";
      else if (op == DFM_FILL_NULL)
        text += valid + "=" + av + "||" + bv + ";" + z + "=" + av + "?" + a +
                ":" + b + ";";
      else if (op == DFM_IS_NULL || op == DFM_IS_NOT_NULL)
        text += z + "=T(" + (op == DFM_IS_NULL ? "!" : "") + av + ");";
      else if (op == DFM_AND || op == DFM_OR) {
        bool isAnd = op == DFM_AND;
        auto symbol = isAnd ? "&&" : "||";
        auto decisive = isAnd ? "==" : "!=";
        text += z + "=T((" + a + "!=T(0))" + symbol + "(" + b + "!=T(0)));";
        text += valid + "=(" + av + "&&" + bv + ")||(" + av + "&&" + a +
                decisive + "T(0))||(" + bv + "&&" + b + decisive + "T(0));";
      } else {
        bool unary = op == DFM_NEG || op == DFM_NOT;
        text +=
            valid + "=" + av + (unary ? "" : "&&" + bv) + ";if(" + valid + "){";
        if (op == DFM_ADD || op == DFM_SUB || op == DFM_MUL || op == DFM_NEG) {
          auto name = op == DFM_ADD   ? "add"
                      : op == DFM_SUB ? "sub"
                      : op == DFM_MUL ? "mul"
                                      : "neg";
          text += "if(!checked_" + std::string(name) + "(" + a + "," +
                  (unary ? "" : b + ",") + z +
                  "))atomic_fetch_min_explicit(error," +
                  std::to_string((r.dtype == DFM_FLOAT32 ? 0x80000000UL : 0UL) +
                                 s.start + j + 1) +
                  "u,memory_order_relaxed);";
        } else {
          std::string expr;
          if (op == DFM_NOT)
            expr = "!(" + a + "!=T(0))";
          else {
            auto symbol = op == DFM_GT   ? ">"
                          : op == DFM_EQ ? "=="
                          : op == DFM_LT ? "<"
                          : op == DFM_GE ? ">="
                          : op == DFM_LE ? "<="
                          : op == DFM_NE ? "!="
                                         : "!=";
            expr = op == DFM_XOR ? "((" + a + "!=T(0))!=(" + b + "!=T(0)))"
                                 : a + symbol + b;
          }
          if (r.dtype == DFM_FLOAT32 && op != DFM_NOT && op != DFM_XOR)
            text += "if(!native_operand(" + a + ")||!native_operand(" + b +
                    "))atomic_fetch_min_explicit(error," +
                    std::to_string(0x80000000UL + s.start + j + 1) +
                    "u,memory_order_relaxed);else ";
          text += z + "=T(" + expr + ");";
        }
        text += "}";
      }
      text += "\n";
    }
    auto last = std::to_string(s.nodes - 1);
    auto idx = std::to_string(s.slot) + "UL*meta.z+row";
    text +=
        "x[" + idx + "]=a" + last + ";v[" + idx + "]=uchar(v" + last + ");}\n";
  }
  return text;
}
std::shared_ptr<Pipelines> pipelines(Context &c, const std::string &src,
                                     bool &hit) {
  auto found = c.cache.find(src);
  hit = found != c.cache.end();
  if (hit) {
    found->second->age = ++c.age;
    return found->second;
  }
  NSError *err = nil;
  auto options = [MTLCompileOptions new];
  options.mathMode = MTLMathModeSafe;
  options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
  auto p = std::make_shared<Pipelines>();
  p->library =
      [c.device newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                             options:options
                               error:&err];
  if (!p->library)
    fail("Metal shader compilation: " +
         std::string(err ? err.localizedDescription.UTF8String
                         : "unknown compiler error"));
  for (NSString *name in p->library.functionNames) {
    auto f = [p->library newFunctionWithName:name];
    auto state = [c.device newComputePipelineStateWithFunction:f error:&err];
    if (!state)
      fail("Metal pipeline creation: " +
           std::string(err ? err.localizedDescription.UTF8String
                           : "unknown compiler error"));
    if (state.maxTotalThreadsPerThreadgroup < 256)
      fail("Metal pipeline cannot execute a 256-thread group");
    p->kernels.emplace(name.UTF8String, state);
  }
  p->age = ++c.age;
  if (c.cache.size() >= 64) {
    auto oldest = c.cache.begin();
    for (auto it = c.cache.begin(); it != c.cache.end(); ++it)
      if (it->second->age < oldest->second->age)
        oldest = it;
    c.cache.erase(oldest);
  }
  c.cache.emplace(src, p);
  return p;
}
struct Buffers {
  Context &c;
  std::vector<id<MTLBuffer>> all;
  int64_t requested = 0;
  id<MTLBuffer> make(int64_t bytes) {
    auto b = [c.device newBufferWithLength:std::max<int64_t>(1, bytes)
                                   options:MTLResourceStorageModeShared |
                                           MTLResourceCPUCacheModeDefaultCache];
    if (!b)
      fail("Metal shared buffer allocation failed");
    requested = add(requested, std::max<int64_t>(1, bytes));
    all.push_back(b);
    return b;
  }
};
struct Drain {
  id<MTLCommandBuffer> command;
  bool committed = false, waited = false;
  ~Drain() {
    if (committed && !waited)
      [command waitUntilCompleted];
  }
};
struct GPUOut {
  int64_t slot, kind, minimum, type;
};
struct Partial {
  int64_t total, count;
};
void run(Context &c, const DFMRequest &r, DFMStats &stats) {
  auto shape = validate(r, true);
  stats = {};
  stats.gpu_ns = -1;
  stats.memory = estimate(r, shape);
  int64_t recommended = static_cast<int64_t>(
      std::min<uint64_t>(c.device.recommendedMaxWorkingSetSize, INT64_MAX));
  int64_t allocated = static_cast<int64_t>(
      std::min<uint64_t>(c.device.currentAllocatedSize, INT64_MAX));
  int64_t budget = std::max<int64_t>(0, recommended - allocated);
  if (r.memory_budget_bytes >= 0)
    budget = std::min(budget, r.memory_budget_bytes);
  if (stats.memory.peak_bytes > budget)
    fail("Metal request exceeds memory budget before submission");
  for (int64_t bytes :
       {shape.matrix, shape.validity, shape.bitmap, shape.literal,
        shape.outputBits, shape.partial, mul(r.output_count, 32),
        mul(shape.cap, 4), mul(shape.blocks, 4), int64_t(16)})
    if (uint64_t(bytes) > c.device.maxBufferLength)
      fail("Metal request exceeds maximum buffer length before submission");
  auto src = source(r);
  bool hit = false;
  auto start = Clock::now();
  auto p = pipelines(c, src, hit);
  stats.pipeline_cache_hit = hit;
  stats.pipeline_compile_ns = hit ? 0 : ns(start);
  Buffers b{c, {}};
  auto x = b.make(shape.matrix), v = b.make(shape.validity);
  id<MTLBuffer> y = nil, w = nil, rank = nil, blocks = nil;
  if (shape.filter) {
    y = b.make(shape.matrix);
    w = b.make(shape.validity);
    rank = b.make(mul(shape.cap, 4));
    blocks = b.make(mul(shape.blocks, 4));
  }
  auto bitmap = b.make(shape.bitmap), literal = b.make(shape.literal),
       bits = b.make(shape.outputBits), partial = b.make(shape.partial);
  auto out = b.make(mul(r.output_count, 32)), meta = b.make(16),
       arithmeticError = b.make(4);
  start = Clock::now();
  auto *metadata = static_cast<uint32_t *>(meta.contents);
  metadata[0] = r.rows;
  metadata[1] = r.rows;
  metadata[2] = shape.cap;
  *static_cast<uint32_t *>(arithmeticError.contents) = UINT32_MAX;
  std::vector<std::array<uint64_t, 4>> unpack;
  int64_t offset = 0;
  for (int64_t i = 0; i < r.input_count; ++i) {
    auto &input = r.inputs[i];
    int64_t size = (r.rows + input.bit_offset + 7) / 8;
    std::array<uint64_t, 4> params{
        static_cast<uint64_t>(i), static_cast<uint64_t>(input.bit_offset),
        static_cast<uint64_t>(input.dtype == DFM_BOOL), UINT64_MAX};
    // Bitmap offsets are absolute; buffer bindings stay naturally aligned.
    if (input.dtype == DFM_BOOL)
      params[2] = offset + 1;
    if (input.dtype == DFM_BOOL) {
      if (size)
        memcpy(static_cast<char *>(bitmap.contents) + offset, input.values,
               size);
      offset += size;
    } else if (r.rows)
      memcpy(static_cast<char *>(x.contents) + i * shape.cap * width(r.dtype),
             input.values, r.rows * width(r.dtype));
    if (input.has_validity) {
      params[3] = offset;
      if (size)
        memcpy(static_cast<char *>(bitmap.contents) + offset, input.validity,
               size);
      offset += size;
    }
    unpack.push_back(params);
  }
  for (int64_t i = 0; i < r.literal_count; ++i) {
    int64_t integer;
    double floating;
    memcpy(&integer, r.literals + i, 8);
    memcpy(&floating, r.literals + i, 8);
    auto op = r.code[4 * i];
    if (r.dtype == DFM_FLOAT32) {
      float value = op == DFM_LITERAL_INT ? float(integer) : float(floating);
      memcpy(static_cast<char *>(literal.contents) + i * 4, &value, 4);
    } else {
      int64_t value = op == DFM_LITERAL_BOOL ? int64_t(floating) : integer;
      if (op == DFM_LITERAL_INT && r.dtype == DFM_INT32 &&
          (value < INT32_MIN || value > INT32_MAX))
        fail("Metal Int32 literal overflow");
      if (r.dtype == DFM_INT32) {
        int32_t narrow = value;
        memcpy(static_cast<char *>(literal.contents) + i * 4, &narrow, 4);
      } else
        memcpy(static_cast<char *>(literal.contents) + i * 8, &value, 8);
    }
  }
  auto *descriptors = static_cast<GPUOut *>(out.contents);
  for (int64_t i = 0; i < r.output_count; ++i)
    descriptors[i] = {r.outputs[i].slot, r.outputs[i].reduction,
                      r.outputs[i].min_count, r.outputs[i].dtype};
  stats.staging_copy_ns = ns(start);
  Drain drain{[c.queue commandBuffer]};
  if (!drain.command)
    fail("Metal command buffer creation failed");
  auto encoder = [drain.command computeCommandEncoder];
  if (!encoder)
    fail("Metal encoder creation failed");
  int64_t launches = 0;
  auto buffer = [&](int idx, id<MTLBuffer> value, int64_t at = 0) {
    [encoder setBuffer:value offset:at atIndex:idx];
  };
  auto bytes = [&](int idx, const void *value, size_t size) {
    [encoder setBytes:value length:size atIndex:idx];
  };
  auto dispatch = [&](const std::string &name, int64_t count,
                      int64_t columns = 1, bool full = false) {
    [encoder setComputePipelineState:p->kernels.at(name)];
    if (full)
      [encoder dispatchThreadgroups:MTLSizeMake((count + 255) / 256, columns, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    else
      [encoder dispatchThreads:MTLSizeMake(std::max<int64_t>(1, count), columns,
                                           1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++launches;
  };
  offset = 0;
  for (int64_t i = 0; i < r.input_count; ++i) {
    buffer(0, x);
    buffer(1, v);
    buffer(2, bitmap);
    bytes(3, unpack[i].data(), 32);
    buffer(4, meta);
    dispatch("unpack", shape.cap);
    auto &input = r.inputs[i];
    offset += ((r.rows + input.bit_offset + 7) / 8) *
              ((input.dtype == DFM_BOOL) + input.has_validity);
  }
  for (int64_t i = 0; i < r.step_count; ++i) {
    auto &step = r.steps[i];
    buffer(0, x);
    buffer(1, v);
    buffer(2, literal);
    buffer(3, meta);
    buffer(4, arithmeticError);
    dispatch("project" + std::to_string(i), shape.cap);
    if (step.filter) {
      uint64_t slot = step.slot;
      buffer(0, x);
      buffer(1, v);
      buffer(2, rank);
      buffer(3, blocks);
      buffer(4, meta);
      bytes(5, &slot, 8);
      dispatch("rank_rows", shape.cap, 1, true);
      buffer(0, blocks);
      buffer(1, meta);
      dispatch("block_offsets", 1);
      for (int64_t j = 0; j < step.gather_count; ++j) {
        uint64_t params[2] = {
            slot, static_cast<uint64_t>(r.gathers[step.gather_start + j])};
        buffer(0, x);
        buffer(1, v);
        buffer(2, y);
        buffer(3, w);
        buffer(4, rank);
        buffer(5, blocks);
        buffer(6, meta);
        bytes(7, params, 16);
        dispatch("gather_rows", shape.cap);
      }
      std::swap(x, y);
      std::swap(v, w);
    }
  }
  uint64_t groups = std::min<int64_t>(shape.blocks, 1024);
  if (r.reductions) {
    buffer(0, x);
    buffer(1, v);
    buffer(2, partial);
    buffer(3, out);
    buffer(4, meta);
    bytes(5, &groups, 8);
    dispatch("reduce_rows", groups * 256, r.output_count, true);
  }
  id<MTLBuffer> result = nil;
  if (r.reductions) {
    result = b.make(mul(r.output_count, 16));
    buffer(0, partial);
    buffer(1, result);
    buffer(2, out);
    buffer(3, meta);
    bytes(4, &groups, 8);
    dispatch("finish_reduce", r.output_count);
  } else {
    uint64_t params[4] = {
        static_cast<uint64_t>(shape.packed),
        static_cast<uint64_t>(r.output_count),
        r.limit < 0 ? UINT64_MAX : static_cast<uint64_t>(r.limit), 0};
    buffer(0, x);
    buffer(1, v);
    buffer(2, bits);
    buffer(3, out);
    buffer(4, meta);
    bytes(5, params, 32);
    dispatch("pack_rows", shape.packed, r.output_count);
  }
  if (b.requested != stats.memory.shared_bytes)
    fail("Metal internal allocation accounting mismatch");
  [encoder endEncoding];
  start = Clock::now();
  [drain.command commit];
  drain.committed = true;
  [drain.command waitUntilCompleted];
  drain.waited = true;
  stats.submit_wait_ns = ns(start);
  stats.synchronizations = 1;
  stats.memory.launches = launches;
  if (drain.command.status != MTLCommandBufferStatusCompleted)
    fail("Metal execution failed: " +
         std::string(drain.command.error
                         ? drain.command.error.localizedDescription.UTF8String
                         : "unknown command buffer error"));
  if (drain.command.GPUEndTime > drain.command.GPUStartTime &&
      drain.command.GPUStartTime > 0)
    stats.gpu_ns =
        int64_t((drain.command.GPUEndTime - drain.command.GPUStartTime) * 1e9);
  uint32_t fault = *static_cast<uint32_t *>(arithmeticError.contents);
  if (fault != UINT32_MAX) {
    if (fault & 0x80000000u)
      fail("Metal unsupported [precision]: Float32 subnormal arithmetic or "
           "underflow requires CPU execution");
    fail("Integer overflow in Metal row expression");
  }
  start = Clock::now();
  int64_t rows = r.reductions ? 1 : metadata[0];
  if (r.limit >= 0)
    rows = std::min(rows, r.limit);
  stats.output_rows = rows;
  // Validate every aggregate before modifying any caller output.
  if (r.reductions && rows) {
    auto *values = static_cast<Partial *>(result.contents);
    for (int64_t i = 0; i < r.output_count; ++i)
      if (r.outputs[i].reduction == DFM_SUM &&
          values[i].count >= r.outputs[i].min_count &&
          (values[i].total < INT32_MIN || values[i].total > INT32_MAX))
        fail("Integer overflow in Metal Int32 sum");
  }
  for (int64_t i = 0; i < r.output_count; ++i) {
    auto &o = r.outputs[i];
    if (r.reductions) {
      if (!rows)
        continue;
      auto a = static_cast<Partial *>(result.contents)[i];
      bool valid = o.reduction != DFM_SUM || a.count >= o.min_count;
      if (o.dtype == DFM_INT32) {
        int32_t value = valid ? a.total : 0;
        memcpy(o.values, &value, 4);
      } else
        memcpy(o.values, &a.total, 8);
      o.validity[0] = valid;
      stats.download_bytes += width(o.dtype) + 1;
    } else {
      int64_t packed = (rows + 7) / 8;
      auto *base =
          static_cast<uint8_t *>(bits.contents) + (2 * i) * shape.packed;
      if (packed)
        memcpy(o.validity, base, packed);
      int64_t size = o.dtype == DFM_BOOL ? packed : rows * width(o.dtype);
      if (size)
        memcpy(o.values,
               o.dtype == DFM_BOOL ? base + shape.packed
                                   : static_cast<uint8_t *>(x.contents) +
                                         o.slot * shape.cap * width(r.dtype),
               size);
      stats.download_bytes += packed + size;
    }
  }
  stats.result_copy_ns = ns(start);
}
} // namespace
extern "C" {
int64_t dfm_abi_version() { return DFM_ABI_VERSION; }
const char *dfm_build_id() { return DFM_BUILD_ID; }
int64_t dfm_device_count() {
  @autoreleasepool {
    return MTLCopyAllDevices().count;
  }
}
void *dfm_context_create(int64_t device_id, int64_t reuse_default, char **err) {
  if (err)
    *err = nullptr;
  @autoreleasepool {
    try {
      std::lock_guard<std::mutex> lock(defaultMutex);
      if (reuse_default && device_id == 0 && defaultContext) {
        ++defaultContext->refs;
        return defaultContext;
      }
      auto devices = MTLCopyAllDevices();
      if (device_id < 0 || device_id >= int64_t(devices.count))
        fail("Metal device index unavailable");
      auto ctx = std::make_unique<Context>();
      ctx->device = devices[device_id];
      ctx->queue = [ctx->device newCommandQueue];
      if (!ctx->queue)
        fail("Metal command queue unavailable");
      ctx->name = ctx->device.name.UTF8String;
      auto *ptr = ctx.release();
      if (reuse_default && device_id == 0) {
        defaultContext = ptr;
        ++ptr->refs;
      }
      return ptr;
    } catch (const std::exception &e) {
      error(err, e);
      return nullptr;
    }
  }
}
void dfm_context_retain(void *p) {
  if (p)
    ++static_cast<Context *>(p)->refs;
}
void dfm_context_release(void *p) {
  if (p && --static_cast<Context *>(p)->refs == 0)
    delete static_cast<Context *>(p);
}
int32_t dfm_context_info(void *p, DFMDeviceInfo *info, char **err) {
  if (err)
    *err = nullptr;
  @autoreleasepool {
    try {
      if (!p || !info)
        fail("Missing Metal context or device info");
      auto &c = *static_cast<Context *>(p);
      *info = {c.name.c_str(),
               c.device.registryID,
               int64_t(c.device.recommendedMaxWorkingSetSize),
               int64_t(c.device.currentAllocatedSize),
               int64_t(c.device.maxThreadsPerThreadgroup.width),
               int64_t(c.device.hasUnifiedMemory)};
      return 0;
    } catch (const std::exception &e) {
      error(err, e);
      return 1;
    }
  }
}
int32_t dfm_estimate(const DFMRequest *r, DFMMemory *m, char **err) {
  if (err)
    *err = nullptr;
  try {
    if (!r || !m)
      fail("Missing Metal estimate request");
    *m = estimate(*r, validate(*r));
    return 0;
  } catch (const std::exception &e) {
    error(err, e);
    return 1;
  }
}
int32_t dfm_plan_cached(void *p, const DFMRequest *r, char **err) {
  if (err)
    *err = nullptr;
  try {
    if (!p || !r)
      fail("Missing Metal cache request");
    validate(*r);
    auto &c = *static_cast<Context *>(p);
    auto src = source(*r);
    std::lock_guard<std::mutex> lock(c.mutex);
    return c.cache.count(src) ? 1 : 0;
  } catch (const std::exception &e) {
    error(err, e);
    return -1;
  }
}
int32_t dfm_execute(void *p, const DFMRequest *r, DFMStats *stats, char **err) {
  if (err)
    *err = nullptr;
  @autoreleasepool {
    try {
      if (!p || !r || !stats)
        fail("Missing Metal execution request");
      auto &c = *static_cast<Context *>(p);
      std::lock_guard<std::mutex> lock(c.mutex);
      run(c, *r, *stats);
      return 0;
    } catch (const std::exception &e) {
      error(err, e);
      return 1;
    }
  }
}
void dfm_free(void *p) { free(p); }
}
