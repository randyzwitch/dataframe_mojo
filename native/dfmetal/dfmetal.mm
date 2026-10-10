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
static_assert(sizeof(DFMOutput) == 48 && sizeof(DFMRequest) == 176);
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
bool binaryOp(int64_t op) {
  return op == DFM_ADD || op == DFM_SUB || op == DFM_MUL || op == DFM_GT ||
         op == DFM_EQ ||
         ((op >= DFM_LT && op <= DFM_NE) ||
          (op >= DFM_FLOORDIV && op <= DFM_CLIP_HIGH)) ||
         (op >= DFM_AND && op <= DFM_KEEP_NULLS);
}
bool unaryOp(int64_t op) {
  return op == DFM_CAST || op == DFM_NEG || op == DFM_ABS || op == DFM_FLOOR ||
         op == DFM_CEIL || op == DFM_ROUND ||
         (op >= DFM_NOT && op <= DFM_IS_INFINITE);
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
  if (t == DFM_FLOAT32 || t == DFM_INT32 || t == DFM_UINT32)
    return 4;
  if (t == DFM_INT64 || t == DFM_UINT64)
    return 8;
  if (t == DFM_INT16 || t == DFM_UINT16)
    return 2;
  if (t == DFM_BOOL || t == DFM_INT8 || t == DFM_UINT8)
    return 1;
  fail("Metal unsupported dtype");
  return 0;
}
bool unsignedType(int64_t t) {
  return t == DFM_UINT8 || t == DFM_UINT16 || t == DFM_UINT32 ||
         t == DFM_UINT64;
}
bool sumSupported(int64_t t) {
  return t != DFM_FLOAT32 && t != DFM_INT64 && t != DFM_UINT64 && t != DFM_BOOL;
}
int64_t sumType(int64_t t) {
  return t == DFM_INT32 || t == DFM_UINT32 ? t : DFM_INT64;
}
bool typed(const DFMRequest &r) { return r.dtype == 0; }
int64_t storageWidth(const DFMRequest &r) {
  return typed(r) ? 8 : width(r.dtype);
}
int64_t nodeType(const DFMRequest &r, int64_t node) {
  return typed(r) ? r.node_types[node] : r.dtype;
}
int64_t slotType(const DFMRequest &r, int64_t slot) {
  return typed(r) ? r.slot_types[slot] : r.dtype;
}
const char *metalType(int64_t t) {
  switch (t) {
  case DFM_FLOAT32:
    return "float";
  case DFM_INT32:
    return "int";
  case DFM_INT64:
    return "long";
  case DFM_BOOL:
  case DFM_UINT8:
    return "uchar";
  case DFM_INT8:
    return "char";
  case DFM_INT16:
    return "short";
  case DFM_UINT16:
    return "ushort";
  case DFM_UINT32:
    return "uint";
  case DFM_UINT64:
    return "ulong";
  default:
    fail("Metal unsupported expression dtype");
    return "";
  }
}
std::string decode(int64_t t, const std::string &word) {
  return t == DFM_FLOAT32 ? "as_type<float>(uint(" + word + "))"
                          : std::string(metalType(t)) + "(" + word + ")";
}
std::string encode(int64_t t, const std::string &value) {
  return t == DFM_FLOAT32 ? "ulong(as_type<uint>(" + value + "))"
                          : "ulong(" + value + ")";
}
int64_t aligned(int64_t n) { return mul(add(n, 7) / 8, 8); }
// A segment ends at a stable filter boundary. Values produced and consumed
// inside one segment stay in per-thread variables; only boundary-live slots
// occupy shared memory. Logical ABI slot identifiers remain unchanged.
struct Layout {
  std::array<int64_t, 64> physical;
  std::vector<int64_t> ends;
  int64_t slots = 0;
  explicit Layout(const DFMRequest &r) {
    physical.fill(-1);
    auto keep = [&](int64_t slot) {
      if (physical[slot] < 0)
        physical[slot] = slots++;
    };
    for (int64_t i = 0; i < r.input_count; ++i)
      keep(i);
    for (int64_t first = 0; first < r.step_count;) {
      if (r.steps[first].filter == 2) {
        ends.push_back(first);
        for (int64_t j = 0; j < r.steps[first].nodes; ++j)
          keep(r.code[4 * (r.steps[first].start + j) + 1]);
        ++first;
        continue;
      }
      int64_t end = first;
      while (end + 1 < r.step_count && !r.steps[end].filter &&
             r.steps[end + 1].filter != 2)
        ++end;
      ends.push_back(end);
      std::array<bool, 64> local{};
      for (int64_t k = first; k <= end; ++k) {
        for (int64_t j = 0; j < r.steps[k].nodes; ++j) {
          auto *c = r.code + 4 * (r.steps[k].start + j);
          if (c[0] == DFM_COLUMN && !local[c[3]])
            keep(c[3]);
        }
        local[r.steps[k].slot] = true;
      }
      if (r.steps[end].filter)
        keep(r.steps[end].slot);
      first = end + 1;
    }
    for (int64_t i = 0; i < r.output_count; ++i)
      keep(r.outputs[i].slot);
  }
};
struct Shape {
  int64_t cap, packed, blocks, bitmap, matrix, validity, literal, outputBits,
      partial;
  bool filter, order;
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
  if (typed(r)) {
    if ((r.literal_count && !r.node_types) || !r.slot_types)
      fail("Missing typed Metal descriptors");
    for (int64_t i = 0; i < r.slots; ++i)
      width(r.slot_types[i]);
    for (int64_t i = 0; i < r.literal_count; ++i)
      width(r.node_types[i]);
  } else
    width(r.dtype);
  if (r.dtype == DFM_BOOL)
    fail("Metal common storage must be numeric");
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
    if (typed(r) ? v.dtype != r.slot_types[i]
                 : (v.dtype != DFM_BOOL && v.dtype != r.dtype))
      fail("Mixed Metal input dtypes");
    if (v.bit_offset < 0 || v.bit_offset > 7 ||
        (v.has_validity != 0 && v.has_validity != 1))
      fail("Invalid input bitmap descriptor");
    if (storage && r.rows && (!v.values || (v.has_validity && !v.validity)))
      fail("Missing Metal input data");
    int64_t bytes = (r.rows + v.bit_offset + 7) / 8;
    if (v.dtype == DFM_BOOL)
      s.bitmap = add(s.bitmap, bytes);
    else if (typed(r))
      s.bitmap = add(aligned(s.bitmap), mul(r.rows, width(v.dtype)));
    if (v.has_validity)
      s.bitmap = add(s.bitmap, bytes);
  }
  std::array<bool, 64> defined{};
  for (int64_t i = 0; i < r.input_count; ++i)
    defined[i] = true;
  for (int64_t i = 0; i < r.step_count; ++i) {
    const auto &v = r.steps[i];
    if (v.start < 0 || v.nodes < 1 || v.nodes > 64 ||
        add(v.start, v.nodes) > r.literal_count ||
        (v.filter != 2 && v.slot < r.input_count) || v.slot < 0 ||
        v.slot >= r.slots || v.gather_start < 0 || v.gather_count < 0 ||
        add(v.gather_start, v.gather_count) > r.gather_count ||
        (v.filter < 0 || v.filter > 2))
      fail("Invalid Metal expression step");
    if (v.filter == 2) {
      if (!typed(r) || v.slot != 0)
        fail("Metal sort requires typed storage");
      for (int64_t j = 0; j < v.nodes; ++j) {
        auto c = r.code + 4 * (v.start + j);
        if (c[0] != DFM_SORT_KEY || c[1] < 0 || c[1] >= r.slots ||
            !defined[c[1]] || c[2] < 0 || c[2] > 1 || c[3] < 0 || c[3] > 1)
          fail("Invalid Metal sort key descriptor");
        if (r.node_types[v.start + j] != slotType(r, c[1]))
          fail("Metal sort key dtype mismatch");
      }
      for (int64_t j = 0; j < v.gather_count; ++j) {
        auto slot = r.gathers[v.gather_start + j];
        if (slot < 0 || slot >= r.slots || !defined[slot])
          fail("Metal sort gathers an undefined slot");
      }
      s.order = true;
      continue;
    }
    s.filter |= v.filter == 1;
    for (int64_t j = 0; j < v.nodes; ++j) {
      const auto *c = r.code + 4 * (v.start + j);
      bool binary = binaryOp(c[0]);
      bool unary = unaryOp(c[0]);
      bool conditional = c[0] == DFM_WHEN;
      bool leaf = c[0] == DFM_COLUMN || c[0] == DFM_LITERAL_INT ||
                  c[0] == DFM_LITERAL_FLOAT || c[0] == DFM_LITERAL_BOOL ||
                  c[0] == DFM_LITERAL_NULL;
      if ((!binary && !unary && !leaf && !conditional) ||
          ((binary || unary || conditional) && (c[1] < 0 || c[1] >= j)) ||
          ((binary || conditional) && (c[2] < 0 || c[2] >= j)) ||
          (conditional && (c[3] < -1 || c[3] >= j)) ||
          (c[0] == DFM_COLUMN && (c[3] < 0 || c[3] >= v.slot)))
        fail("Invalid Metal expression bytecode");
      if (c[0] == DFM_CAST && (!typed(r) || (c[3] != 0 && c[3] != 1)))
        fail("Invalid Metal cast descriptor");
      if (c[0] == DFM_COLUMN && !defined[c[3]])
        fail("Metal expression references an undefined slot");
      auto operand = (binary || unary || conditional)
                         ? nodeType(r, v.start + c[1])
                         : nodeType(r, v.start + j);
      if ((c[0] >= DFM_IS_NAN && c[0] <= DFM_IS_INFINITE) &&
          operand != DFM_FLOAT32)
        fail("Metal floating classification requires Float32");
      if ((c[0] == DFM_FLOORDIV || c[0] == DFM_MOD || c[0] == DFM_POW) &&
          operand == DFM_FLOAT32)
        fail("Metal floating operator requires CPU execution");
      if (c[0] == DFM_ROUND && operand == DFM_FLOAT32 && c[3] != 0)
        fail("Metal Float32 decimal rounding requires CPU execution");
      if (nodeType(r, v.start + j) != DFM_FLOAT32 && c[0] == DFM_LITERAL_FLOAT)
        fail("Floating literal requires native Float32 storage");
    }
    if (typed(r)) {
      if (r.slot_types[v.slot] != r.node_types[v.start + v.nodes - 1])
        fail("Metal step result dtype mismatch");
      if (v.filter && r.slot_types[v.slot] != DFM_BOOL)
        fail("Metal filter requires Bool");
      for (int64_t j = 0; j < v.nodes; ++j) {
        const auto *c = r.code + 4 * (v.start + j);
        auto t = r.node_types[v.start + j];
        if (c[0] == DFM_COLUMN && t != r.slot_types[c[3]])
          fail("Metal column dtype mismatch");
        if (c[0] == DFM_LITERAL_BOOL && t != DFM_BOOL)
          fail("Metal Bool literal dtype mismatch");
        bool compare = (c[0] >= DFM_GT && c[0] <= DFM_EQ) ||
                       (c[0] >= DFM_LT && c[0] <= DFM_NE);
        bool logic = c[0] == DFM_AND || c[0] == DFM_OR || c[0] == DFM_XOR ||
                     c[0] == DFM_NOT;
        bool nulltest = c[0] == DFM_IS_NULL || c[0] == DFM_IS_NOT_NULL ||
                        (c[0] >= DFM_IS_NAN && c[0] <= DFM_IS_INFINITE);
        bool math = c[0] == DFM_ADD || c[0] == DFM_SUB || c[0] == DFM_MUL ||
                    c[0] == DFM_NEG || c[0] == DFM_FILL_NULL ||
                    c[0] == DFM_ABS || c[0] == DFM_FLOOR || c[0] == DFM_CEIL ||
                    c[0] == DFM_ROUND || c[0] == DFM_FLOORDIV ||
                    c[0] == DFM_MOD || c[0] == DFM_POW ||
                    c[0] == DFM_CLIP_LOW || c[0] == DFM_CLIP_HIGH ||
                    c[0] == DFM_FILL_NAN;
        bool unaryMath = c[0] == DFM_NEG || c[0] == DFM_ABS ||
                         c[0] == DFM_FLOOR || c[0] == DFM_CEIL ||
                         c[0] == DFM_ROUND;
        if (c[0] == DFM_WHEN &&
            (r.node_types[v.start + c[1]] != DFM_BOOL ||
             t != r.node_types[v.start + c[2]] ||
             (c[3] >= 0 && t != r.node_types[v.start + c[3]])))
          fail("Metal conditional dtype mismatch");
        if (c[0] == DFM_KEEP_NULLS && t != r.node_types[v.start + c[2]])
          fail("Metal validity selection dtype mismatch");
        if (c[0] == DFM_FILL_NAN && t != DFM_FLOAT32)
          fail("Metal fill NaN requires Float32");
        if ((compare || logic || nulltest) && t != DFM_BOOL)
          fail("Metal predicate dtype mismatch");
        if (logic &&
            (r.node_types[v.start + c[1]] != DFM_BOOL ||
             (c[0] != DFM_NOT && r.node_types[v.start + c[2]] != DFM_BOOL)))
          fail("Metal Boolean operand dtype mismatch");
        if ((compare || math) &&
            (r.node_types[v.start + c[1]] !=
                 (math ? t : r.node_types[v.start + c[2]]) ||
             (!unaryMath &&
              r.node_types[v.start + c[1]] != r.node_types[v.start + c[2]])))
          fail("Metal numeric operand dtype mismatch");
      }
    }
    defined[v.slot] = true;
    for (int64_t j = 0; j < v.gather_count; ++j) {
      auto slot = r.gathers[v.gather_start + j];
      if (slot < 0 || slot >= r.slots || !defined[slot])
        fail("Metal gather references an undefined slot");
    }
  }
  for (int64_t i = 0; i < r.gather_count; ++i)
    if (r.gathers[i] < 0 || r.gathers[i] >= r.slots)
      fail("Invalid Metal gather slot");
  for (int64_t i = 0; i < r.output_count; ++i) {
    const auto &v = r.outputs[i];
    width(v.dtype);
    if (v.slot < 0 || v.slot >= r.slots || v.min_count < 0 || !defined[v.slot])
      fail("Invalid Metal output descriptor");
    if (!r.reductions) {
      if (v.reduction != -1 ||
          (typed(r) ? v.dtype != r.slot_types[v.slot]
                    : (v.dtype != r.dtype && v.dtype != DFM_BOOL)))
        fail("Invalid Metal row output");
    } else {
      if (v.reduction != DFM_COUNT && v.reduction != DFM_LEN &&
          !(v.reduction == DFM_SUM && sumSupported(slotType(r, v.slot))) &&
          !((v.reduction == DFM_MIN || v.reduction == DFM_MAX) && typed(r)))
        fail("Metal reduction requires unsupported accumulator precision");
      if (v.dtype != (v.reduction == DFM_SUM ? sumType(slotType(r, v.slot))
                      : (v.reduction == DFM_MIN || v.reduction == DFM_MAX)
                          ? slotType(r, v.slot)
                          : DFM_INT64))
        fail("Invalid Metal reduction dtype");
    }
    if (storage && (r.rows || r.reductions) && (!v.values || !v.validity))
      fail("Missing Metal output storage");
  }
  auto layout = Layout(r);
  s.matrix = mul(mul(s.cap, layout.slots), storageWidth(r));
  s.validity = mul(s.cap, layout.slots);
  s.literal = std::max<int64_t>(1, mul(r.literal_count, storageWidth(r)));
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
  if (s.filter || s.order) {
    account(s.matrix);
    account(s.validity);
    account(mul(s.cap, 4));
    account(mul(s.order ? s.cap : s.blocks, 4));
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
    const auto &input = r.inputs[i];
    auto bitmapBytes = (r.rows + input.bit_offset + 7) / 8;
    m.upload_bytes = add(m.upload_bytes, input.dtype == DFM_BOOL
                                             ? bitmapBytes
                                             : mul(r.rows, width(input.dtype)));
    if (input.has_validity)
      m.upload_bytes = add(m.upload_bytes, bitmapBytes);
  }
  m.upload_bytes = add(m.upload_bytes, mul(r.literal_count, storageWidth(r)));
  for (int64_t i = 0; i < r.output_count; ++i) {
    auto t = r.outputs[i].dtype;
    m.result_bytes =
        add(m.result_bytes,
            r.reductions ? width(t) + 1
                         : add(t == DFM_BOOL ? s.packed : mul(r.rows, width(t)),
                               s.packed));
  }
  m.peak_bytes = add(m.shared_bytes, m.result_bytes);
  auto layout = Layout(r);
  m.launches = r.input_count + layout.ends.size() + (r.reductions ? 2 : 1);
  for (int64_t i = 0; i < r.step_count; ++i)
    if (r.steps[i].filter) {
      if (r.steps[i].filter == 1)
        m.launches += 2;
      else
        for (int64_t stride = 1; stride < s.cap; stride *= 2)
          ++m.launches;
      for (int64_t j = 0; j < r.steps[i].gather_count; ++j)
        m.launches +=
            layout.physical[r.gathers[r.steps[i].gather_start + j]] >= 0;
    }
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
std::string castExpression(int64_t from, int64_t to, const std::string &a,
                           const std::string &z, const std::string &valid,
                           bool strict, int64_t id) {
  auto target = std::string(metalType(to));
  std::string body = "bool fits=true;";
  if (to == DFM_BOOL) {
    if (from == DFM_FLOAT32)
      body += "uint ab=as_type<uint>(" + a +
              ")&0x7fffffffu;fits=ab<=0x7f800000u;" + z + "=uchar(ab!=0);";
    else
      body += z + "=uchar(" + a + "!=0);";
  } else if (to == DFM_FLOAT32) {
    if (from == DFM_INT64 || from == DFM_UINT64) {
      auto mag = from == DFM_INT64
                     ? "(" + a + "<0?0UL-ulong(" + a + "):ulong(" + a + "))"
                     : "ulong(" + a + ")";
      // Native Float32 conversion is exact for the CPU contract unless the
      // CPU's intervening Float64 rounding can land on a Float32 midpoint.
      // Detect that narrow region; never emulate Float64 on the device.
      body += "ulong mag=" + mag +
              ";if(mag>9007199254740992UL){uint e=63-clz(mag);ulong "
              "step=1UL<<(e-23),rest=mag&(step-1),midpoint=step/"
              "2,dist=rest>midpoint?rest-midpoint:midpoint-rest;if(dist!=0&&"
              "dist<=(1UL<<(e-53)))atomic_fetch_min_explicit(error," +
              std::to_string(0x20000000UL + id) + "u,memory_order_relaxed);}";
    }
    body += z + "=float(" + a + ");";
  } else if (from == DFM_FLOAT32) {
    auto bits = width(to) * 8 - (unsignedType(to) ? 0 : 1);
    // Powers of two are exactly representable in Float32, including 2**64.
    body += "float whole=trunc(" + a + ");float upper=0x1p" +
            std::to_string(bits) + "f;fits=isfinite(" + a +
            ")&&whole<upper&&whole>=" +
            (unsignedType(to) ? std::string("0.0f") : std::string("-upper")) +
            ";if(fits)" + z + "=" + target + "(whole);";
  } else {
    auto bits = width(to) * 8;
    auto umax = bits == 64 ? std::string("18446744073709551615UL")
                           : std::to_string((uint64_t(1) << bits) - 1) + "UL";
    auto smax = bits == 64
                    ? std::string("9223372036854775807L")
                    : std::to_string((uint64_t(1) << (bits - 1)) - 1) + "L";
    auto smin = bits == 64 ? std::string("(-9223372036854775807L-1L)")
                           : "(-" + smax + "-1L)";
    if (unsignedType(to))
      body += "fits=" +
              (unsignedType(from) || from == DFM_BOOL ? std::string()
                                                      : a + ">=0&&") +
              "ulong(" + a + ")<=" + umax + ";";
    else if (unsignedType(from) || from == DFM_BOOL)
      body += "fits=ulong(" + a + ")<=ulong(" + smax + ");";
    else
      body +=
          "fits=long(" + a + ")>=" + smin + "&&long(" + a + ")<=" + smax + ";";
    body += "if(fits)" + z + "=" + target + "(" + a + ");";
  }
  body += "if(!fits){" + valid + "=false;" + z + "=" + target + "(0);";
  if (strict)
    body += "atomic_fetch_min_explicit(error," +
            std::to_string(0x40000000UL + id) + "u,memory_order_relaxed);";
  return body + "}";
}
std::string arithmeticSource(int64_t dtype) {
  std::string text = std::string("typedef ") + metalType(dtype) + " T;\n";
  text += R"MSL(
// Unsigned magnitudes avoid signed overflow, including the minimum Int64.
bool checked_add(T a,T b,thread T &z) {
)MSL";
  if (dtype == DFM_FLOAT32) {
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
  } else if (unsignedType(dtype)) {
    std::string hi = dtype == DFM_UINT8    ? "255UL"
                     : dtype == DFM_UINT16 ? "65535UL"
                     : dtype == DFM_UINT32 ? "4294967295UL"
                                           : "18446744073709551615UL";
    text += "const ulong hi=" + hi +
            ";ulong x=a,y=b;if(x>hi-y)return false;z=T(x+y);return true;}\n";
    text += "bool checked_sub(T a,T b,thread T &z){if(a<b)return "
            "false;z=T(a-b);return true;}\n";
    text += "bool checked_mul(T a,T b,thread T &z){const ulong hi=" + hi +
            ";ulong x=a,y=b,u=x*y;if(mulhi(x,y)!=0||u>hi)return "
            "false;z=T(u);return true;}\n";
    text += "bool checked_neg(T a,thread T &z){if(a)return false;z=0;return "
            "true;}\n";
  } else {
    std::string lo = dtype == DFM_INT8    ? "(-127L-1L)"
                     : dtype == DFM_INT16 ? "(-32767L-1L)"
                     : dtype == DFM_INT32 ? "(-2147483647L-1L)"
                                          : "(-9223372036854775807L-1L)";
    std::string hi = dtype == DFM_INT8    ? "127L"
                     : dtype == DFM_INT16 ? "32767L"
                     : dtype == DFM_INT32 ? "2147483647L"
                                          : "9223372036854775807L";
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
  if (dtype != DFM_FLOAT32) {
    text += "bool checked_divmod(T a,T b,thread T &z,bool modulo){";
    if (unsignedType(dtype))
      text += "z=modulo?T(a%b):T(a/b);return true;}\n";
    else
      text += "if(b==T(-1)){if(modulo){z=0;return true;}return "
              "checked_neg(a,z);}long "
              "x=a,y=b,q=x/"
              "y,r=x%y;if(r&&((r<0)!=(y<0))){--q;r+=y;}z=T(modulo?r:q);return "
              "true;}\n";
    text += "bool checked_pow(T a,T b,thread T &z){";
    if (!unsignedType(dtype))
      text += "if(b<0)return false;";
    text += "ulong n=ulong(b);T "
            "result=T(1),factor=a;while(n){if((n&1)&&!checked_mul(result,"
            "factor,result))return "
            "false;n>>=1;if(n&&!checked_mul(factor,factor,factor))return "
            "false;}z=result;return true;}\n";
  }
  return text;
}
std::string source(const DFMRequest &r) {
  auto layout = Layout(r);
  std::string text = "#include <metal_stdlib>\nusing namespace metal;\n#pragma "
                     "clang fp contract(off)\n";
  text += "struct Out { long slot; long kind; long minimum; long type; "
          "};\nstruct Partial { long total; long count; };\n";
  text += R"MSL(
bool float_compare(float a,float b,int op){
 uint x=as_type<uint>(a),y=as_type<uint>(b),ax=x&0x7fffffffU,ay=y&0x7fffffffU;
 if(ax>0x7f800000U||ay>0x7f800000U)return op==23;
 bool equal=x==y||(ax==0&&ay==0);uint kx=(x&0x80000000U)?~x:(x^0x80000000U),ky=(y&0x80000000U)?~y:(y^0x80000000U);
 if(op==9)return equal;if(op==23)return !equal;if(op==8)return !equal&&kx>ky;
 if(op==20)return !equal&&kx<ky;if(op==21)return equal||kx>ky;return equal||kx<ky;
}
float float_integral(float x,int op){
 uint bits=as_type<uint>(x),sign=bits&0x80000000U,ab=bits&0x7fffffffU;
 int e=int((bits>>23)&255)-127;if(e>=23||ab==0)return x;
 if(e<0){if(op==55)return as_type<float>(sign?0xbf800000U:sign);
 if(op==56)return as_type<float>(sign?sign:0x3f800000U);
 return as_type<float>(e==-1?(sign|0x3f800000U):sign);}
 uint unit=1U<<(23-e),rest=bits&(unit-1),whole=bits&~(unit-1);
 if(rest&&((op==55&&sign)||(op==56&&!sign)||(op==57&&rest>=unit/2)))whole+=unit;
 return as_type<float>(whole);
}
)MSL";
  if (typed(r)) {
    text += "typedef ulong T;\n";
    std::array<bool, 11> used{};
    for (int64_t i = 0; i < r.literal_count; ++i)
      used[r.node_types[i]] = true;
    for (int64_t t = 1; t <= 10; ++t)
      if (used[t] && t != DFM_BOOL)
        text += "namespace d" + std::to_string(t) + " {\n" +
                arithmeticSource(t) + "}\n";
  } else
    text += arithmeticSource(r.dtype);
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
// Extrema retain the earliest selected row on ties. Partial.count is row+1
// for extrema and the valid count for sum/count; zero denotes no candidate.
// Compare Float32 encodings to preserve subnormal ordering without FTZ.
bool extrema_better(ulong a,ulong b,long type,bool maximum) {
 if(type==1){uint x=uint(a),y=uint(b),ax=x&0x7fffffffU,ay=y&0x7fffffffU;
 bool nx=ax>0x7f800000U,ny=ay>0x7f800000U;
 if(nx||ny)return nx!=ny&&(maximum?nx:ny);
 if(ax==0&&ay==0)return false;
 uint kx=(x&0x80000000U)?~x:(x^0x80000000U);
 uint ky=(y&0x80000000U)?~y:(y^0x80000000U);
 return maximum?kx>ky:kx<ky;}
 if(type==2||type==3||type==5||type==6)
 return maximum?long(a)>long(b):long(a)<long(b);
 return maximum?a>b:a<b;
}
void extrema_merge(thread long &value,thread long &index,long candidate,long other,Out out) {
 if(!other)return;
 bool take=!index||extrema_better(ulong(candidate),ulong(value),out.type,out.kind==81);
 if(!take&&!extrema_better(ulong(value),ulong(candidate),out.type,out.kind==81)&&other<index)take=true;
 if(take){value=candidate;index=other;}
}
kernel void reduce_rows(device const T *x[[buffer(0)]],device const uchar *v[[buffer(1)]],
 device Partial *partials[[buffer(2)]],device const Out *out[[buffer(3)]],
 constant uint4 &meta[[buffer(4)]],constant ulong &groups[[buffer(5)]],
 uint2 group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_threadgroup]]) {
 threadgroup long sums[256];threadgroup long counts[256];long sum=0,count=0;
 Out o=out[group.y];bool extrema=o.kind==80||o.kind==81;
 for(ulong row=group.x*256+lane;row<meta.x;row+=groups*256){ulong idx=o.slot*meta.z+row;
 if(v[idx]){if(extrema)extrema_merge(sum,count,long(x[idx]),long(row+1),o);
 else{++count;if(o.kind==10)sum+=long(x[idx]);}}}
 sums[lane]=sum;counts[lane]=count;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint stride=128;stride;stride/=2){if(lane<stride){
 if(extrema){long a=sums[lane],b=counts[lane];extrema_merge(a,b,sums[lane+stride],counts[lane+stride],o);sums[lane]=a;counts[lane]=b;}
 else{sums[lane]+=sums[lane+stride];counts[lane]+=counts[lane+stride];}}
 threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(lane==0)partials[group.y*groups+group.x]={sums[0],counts[0]};
}
kernel void finish_reduce(device Partial *partials[[buffer(0)]],device Partial *result[[buffer(1)]],
 device const Out *out[[buffer(2)]],constant uint4 &meta[[buffer(3)]],constant ulong &groups[[buffer(4)]],
 uint column[[thread_position_in_grid]]) {
 long sum=0,count=0;Out o=out[column];bool extrema=o.kind==80||o.kind==81;
 for(ulong i=0;i<groups;++i){Partial p=partials[column*groups+i];
 if(extrema)extrema_merge(sum,count,p.total,p.count,o);else{sum+=p.total;count+=p.count;}}
 if(o.kind==11){sum=count;count=1;}if(o.kind==90){sum=meta.x;count=1;}
 result[column]={sum,count};
}
kernel void order_init(device uint *index[[buffer(0)]],constant uint4 &meta[[buffer(1)]],uint row[[thread_position_in_grid]]){
 if(row<meta.x)index[row]=row;
}
kernel void order_gather(device const T *x[[buffer(0)]],device const uchar *v[[buffer(1)]],
 device T *y[[buffer(2)]],device uchar *w[[buffer(3)]],device const uint *index[[buffer(4)]],
 constant uint4 &meta[[buffer(5)]],constant ulong &slot[[buffer(6)]],uint row[[thread_position_in_grid]]){
 if(row>=meta.x)return;ulong target=slot*meta.z+row,source=slot*meta.z+index[row];y[target]=x[source];w[target]=v[source];
}
)MSL";
  if (typed(r))
    for (int64_t i = 0; i < r.input_count; ++i) {
      auto t = r.inputs[i].dtype;
      text += "kernel void unpack" + std::to_string(i) +
              "(device ulong *x[[buffer(0)]],device uchar "
              "*v[[buffer(1)]],device const uchar *bits[[buffer(2)]],constant "
              "ulong4 &p[[buffer(3)]],constant uint4 &meta[[buffer(4)]],uint "
              "row[[thread_position_in_grid]]) {if(row>=meta.x)return;ulong "
              "at=p.x*meta.z+row,bit=row+p.y;";
      if (t == DFM_BOOL)
        text += "x[at]=ulong((bits[p.z-1+bit/8]>>(bit%8))&1);";
      else
        text += "x[at]=" +
                encode(t, "((device const " + std::string(metalType(t)) +
                              " *)(bits+p.z))[row]") +
                ";";
      text += "v[at]=p.w==ulong(-1)?1:uchar((bits[p.w+bit/8]>>(bit%8))&1); }\n";
    }
  for (int64_t i = 0; i < r.step_count; ++i)
    if (r.steps[i].filter == 2) {
      auto &step = r.steps[i];
      auto name = std::to_string(i);
      text += "bool order_less" + name +
              "(uint a,uint b,device const T *x,device const uchar *v,constant "
              "uint4 &meta){";
      for (int64_t j = 0; j < step.nodes; ++j) {
        auto key = r.code + 4 * (step.start + j);
        auto slot = std::to_string(layout.physical[key[1]]) + "UL*meta.z";
        text += "{ulong ai=" + slot + "+a,bi=" + slot +
                "+b;bool av=v[ai]!=0,bv=v[bi]!=0;if(av!=bv)return " +
                std::string(key[3] ? "av" : "!av") + ";if(av){";
        if (slotType(r, key[1]) == DFM_FLOAT32)
          text += "bool "
                  "an=(uint(x[ai])&0x7fffffffU)>0x7f800000U,bn=(uint(x[bi])&"
                  "0x7fffffffU)>0x7f800000U;if(an!=bn)return !an;";
        auto type = std::to_string(slotType(r, key[1])),
             maximum = std::string(key[2] ? "true" : "false");
        text += "if(extrema_better(ulong(x[ai]),ulong(x[bi])," + type + "," +
                maximum +
                "))return true;if(extrema_better(ulong(x[bi]),ulong(x[ai])," +
                type + "," + maximum + "))return false;}}";
      }
      text += "return a<b;}\n";
      text +=
          "kernel void order_merge" + name +
          "(device const T *x[[buffer(0)]],device const uchar "
          "*v[[buffer(1)]],device const uint *index[[buffer(2)]],device uint "
          "*target[[buffer(3)]],constant uint4 &meta[[buffer(4)]],constant "
          "uint &stride[[buffer(5)]],uint "
          "row[[thread_position_in_grid]]){if(row>=meta.x)return;uint "
          "base=(row/"
          "(2*stride))*(2*stride),middle=min(base+stride,meta.x),end=min(base+"
          "2*stride,meta.x);bool left=row<middle;uint "
          "first=left?middle:base,lo=first,hi=left?end:middle,own=index[row];"
          "while(lo<hi){uint mid=(lo+hi)/2;if(order_less" +
          name +
          "(index[mid],own,x,v,meta))lo=mid+1;else "
          "hi=mid;}target[base+(row-(left?base:middle))+(lo-first)]=own;}\n";
    }
  int64_t first = 0;
  for (auto end : layout.ends) {
    if (r.steps[end].filter == 2) {
      first = end + 1;
      continue;
    }
    std::array<bool, 64> local{};
    text += "kernel void project" + std::to_string(end) +
            "(device T *x[[buffer(0)]],device uchar *v[[buffer(1)]],device "
            "const T *lit[[buffer(2)]],constant uint4 "
            "&meta[[buffer(3)]],device atomic_uint *error[[buffer(4)]],uint "
            "row[[thread_position_in_grid]]){if(row>=meta.x)return;\n";
    for (int64_t k = first; k <= end; ++k) {
      auto &s = r.steps[k];
      auto prefix = "s" + std::to_string(k) + "_";
      std::vector<std::string> active(s.nodes);
      active.back() = "true";
      auto observe = [&](int64_t node, const std::string &mask) {
        if (node < 0 || active[node] == "true")
          return;
        if (mask == "true" || active[node].empty())
          active[node] = mask;
        else
          active[node] = "(" + active[node] + "||" + mask + ")";
      };
      for (int64_t j = s.nodes - 1; j >= 0; --j) {
        if (active[j].empty())
          continue;
        auto c = r.code + 4 * (s.start + j);
        if (c[0] == DFM_WHEN) {
          observe(c[1], active[j]);
          auto id = prefix + std::to_string(c[1]);
          auto yes = "(v" + id + "&&a" + id + "!=0)";
          observe(c[2], "(" + active[j] + "&&" + yes + ")");
          observe(c[3], "(" + active[j] + "&&!" + yes + ")");
        } else if (binaryOp(c[0]) || unaryOp(c[0])) {
          observe(c[1], active[j]);
          if (binaryOp(c[0]))
            observe(c[2], active[j]);
        }
      }
      for (int64_t j = 0; j < s.nodes; ++j) {
        auto *c = r.code + 4 * (s.start + j);
        auto op = c[0];
        auto id = prefix + std::to_string(j);
        std::string a = "a" + prefix + std::to_string(c[1]),
                    b = "a" + prefix + std::to_string(c[2]);
        std::string av = "v" + prefix + std::to_string(c[1]),
                    bv = "v" + prefix + std::to_string(c[2]);
        auto dtype = nodeType(r, s.start + j);
        auto typename_ =
            typed(r) ? std::string(metalType(dtype)) : std::string("T");
        text += typename_ + " a" + id + "=" + typename_ + "(0);bool v" + id +
                "=true;";
        std::string z = "a" + id, valid = "v" + id;
        bool masked = active[j] != "true";
        if (masked)
          text += valid + "=false;if(" +
                  (active[j].empty() ? std::string("false") : active[j]) +
                  "){" + valid + "=true;";
        if (op == DFM_COLUMN) {
          if (local[c[3]]) {
            auto slot = std::to_string(c[3]);
            text += z + "=slot" + slot + ";" + valid + "=valid" + slot + ";";
          } else {
            auto idx = std::to_string(layout.physical[c[3]]) + "UL*meta.z+row";
            text += z + "=" +
                    (typed(r) ? decode(dtype, "x[" + idx + "]")
                              : "x[" + idx + "]") +
                    ";" + valid + "=v[" + idx + "]!=0;";
          }
        } else if (op == DFM_CAST) {
          text += valid + "=" + av + ";if(" + valid + "){" +
                  castExpression(r.node_types[s.start + c[1]], dtype, a, z,
                                 valid, c[3] != 0, s.start + j + 1) +
                  "}";
        } else if (op == DFM_LITERAL_NULL)
          text += valid + "=false;";
        else if (op == DFM_LITERAL_INT || op == DFM_LITERAL_FLOAT ||
                 op == DFM_LITERAL_BOOL)
          text += z + "=" +
                  (typed(r) ? decode(dtype,
                                     "lit[" + std::to_string(s.start + j) + "]")
                            : "lit[" + std::to_string(s.start + j) + "]") +
                  ";";
        else if (op == DFM_WHEN) {
          auto yes = "(" + av + "&&" + a + "!=0)";
          auto other = prefix + std::to_string(c[3]);
          text += valid + "=" + yes + "?" + bv + ":" +
                  (c[3] >= 0 ? "v" + other : "false") + ";";
          text += z + "=" + yes + "?" + b + ":" +
                  (c[3] >= 0 ? "a" + other : typename_ + "(0)") + ";";
        } else if (op == DFM_KEEP_NULLS)
          text += valid + "=" + av + "&&" + bv + ";" + z + "=" + b + ";";
        else if (op == DFM_FILL_NAN) {
          auto nan = "((as_type<uint>(" + a + ")&0x7fffffffU)>0x7f800000U)";
          text += valid + "=" + av + "&&(!" + nan + "||" + bv + ");" + z + "=" +
                  nan + "?" + b + ":" + a + ";";
        } else if (op == DFM_FILL_NULL)
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
          bool unary = unaryOp(op);
          text += valid + "=" + av + (unary ? "" : "&&" + bv) + ";if(" + valid +
                  "){";
          if (op >= DFM_IS_NAN && op <= DFM_IS_INFINITE) {
            auto bits = "(as_type<uint>(" + a + ")&0x7fffffffU)";
            auto test = op == DFM_IS_NAN       ? bits + ">0x7f800000U"
                        : op == DFM_IS_NOT_NAN ? bits + "<=0x7f800000U"
                        : op == DFM_IS_FINITE  ? bits + "<0x7f800000U"
                                               : bits + "==0x7f800000U";
            text += z + "=uchar(" + test + ");";
          } else if (op == DFM_FLOOR || op == DFM_CEIL || op == DFM_ROUND) {
            text += z + "=" +
                    (dtype == DFM_FLOAT32 ? "float_integral(" + a + "," +
                                                std::to_string(op) + ")"
                                          : a) +
                    ";";
          } else if (op == DFM_ABS) {
            if (dtype == DFM_FLOAT32)
              text +=
                  z + "=as_type<float>(as_type<uint>(" + a + ")&0x7fffffffU);";
            else if (unsignedType(dtype))
              text += z + "=" + a + ";";
            else
              text += "if(" + a + "<0){if(!" +
                      (typed(r) ? "d" + std::to_string(dtype) + "::"
                                : std::string()) +
                      "checked_neg(" + a + "," + z +
                      "))atomic_fetch_min_explicit(error," +
                      std::to_string(s.start + j + 1) +
                      "u,memory_order_relaxed);}else " + z + "=" + a + ";";
          } else if (op == DFM_FLOORDIV || op == DFM_MOD || op == DFM_POW) {
            auto helper =
                (typed(r) ? "d" + std::to_string(dtype) + "::" : std::string());
            if (op != DFM_POW)
              text += "if(" + b + "==0)" + valid + "=false;else ";
            else if (!unsignedType(dtype))
              text += "if(" + b + "<0)atomic_fetch_min_explicit(error," +
                      std::to_string(0x10000000UL + s.start + j + 1) +
                      "u,memory_order_relaxed);else ";
            text += "if(!" + helper +
                    (op == DFM_POW ? "checked_pow(" : "checked_divmod(") + a +
                    "," + b + "," + z +
                    (op == DFM_POW   ? std::string()
                     : op == DFM_MOD ? ",true"
                                     : ",false") +
                    "))atomic_fetch_min_explicit(error," +
                    std::to_string(s.start + j + 1) +
                    "u,memory_order_relaxed);";
          } else if (op == DFM_CLIP_LOW || op == DFM_CLIP_HIGH) {
            auto compare = dtype == DFM_FLOAT32
                               ? "float_compare(" + a + "," + b + "," +
                                     (op == DFM_CLIP_LOW ? "20" : "8") + ")"
                               : a + (op == DFM_CLIP_LOW ? "<" : ">") + b;
            text += z + "=" + compare + "?" + b + ":" + a + ";";
          } else if (op == DFM_ADD || op == DFM_SUB || op == DFM_MUL ||
                     op == DFM_NEG) {
            auto name = op == DFM_ADD   ? "add"
                        : op == DFM_SUB ? "sub"
                        : op == DFM_MUL ? "mul"
                                        : "neg";
            text += "if(!" +
                    (typed(r) ? "d" + std::to_string(dtype) + "::" : "") +
                    "checked_" + std::string(name) + "(" + a + "," +
                    (unary ? "" : b + ",") + z +
                    "))atomic_fetch_min_explicit(error," +
                    std::to_string((dtype == DFM_FLOAT32 ? 0x80000000UL : 0UL) +
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
            if (nodeType(r, s.start + c[1]) == DFM_FLOAT32 && op != DFM_NOT &&
                op != DFM_XOR)
              expr = "float_compare(" + a + "," + b + "," + std::to_string(op) +
                     ")";
            text += z + "=T(" + expr + ");";
          }
          text += "}";
        }
        if (masked)
          text += "}";
        text += "\n";
      }
      auto last = prefix + std::to_string(s.nodes - 1);
      auto slot = std::to_string(s.slot);
      text +=
          std::string(
              local[s.slot]
                  ? "slot"
                  : (typed(r) ? std::string(metalType(r.slot_types[s.slot])) +
                                    " slot"
                              : "T slot")) +
          slot + "=a" + last + ";" + (local[s.slot] ? "valid" : "bool valid") +
          slot + "=v" + last + ";\n";
      local[s.slot] = true;
      if (layout.physical[s.slot] >= 0) {
        auto idx = std::to_string(layout.physical[s.slot]) + "UL*meta.z+row";
        text += "x[" + idx + "]=" +
                (typed(r) ? encode(r.slot_types[s.slot], "slot" + slot)
                          : "slot" + slot) +
                ";v[" + idx + "]=uchar(valid" + slot + ");\n";
      }
    }
    text += "}\n";
    first = end + 1;
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
  auto layout = Layout(r);
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
        mul(shape.cap, 4), mul(shape.order ? shape.cap : shape.blocks, 4),
        int64_t(16)})
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
  if (shape.filter || shape.order) {
    y = b.make(shape.matrix);
    w = b.make(shape.validity);
    rank = b.make(mul(shape.cap, 4));
    blocks = b.make(mul(shape.order ? shape.cap : shape.blocks, 4));
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
        static_cast<uint64_t>(layout.physical[i]),
        static_cast<uint64_t>(input.bit_offset),
        static_cast<uint64_t>(input.dtype == DFM_BOOL), UINT64_MAX};
    // Bitmap offsets are absolute; buffer bindings stay naturally aligned.
    if (input.dtype == DFM_BOOL)
      params[2] = offset + 1;
    if (input.dtype == DFM_BOOL) {
      if (size)
        memcpy(static_cast<char *>(bitmap.contents) + offset, input.values,
               size);
      offset += size;
    } else if (typed(r)) {
      offset = aligned(offset);
      params[2] = offset;
      if (r.rows)
        memcpy(static_cast<char *>(bitmap.contents) + offset, input.values,
               r.rows * width(input.dtype));
      offset += r.rows * width(input.dtype);
    } else if (r.rows)
      memcpy(static_cast<char *>(x.contents) +
                 layout.physical[i] * shape.cap * storageWidth(r),
             input.values, r.rows * storageWidth(r));
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
    auto dtype = nodeType(r, i);
    if (dtype == DFM_FLOAT32) {
      float value = op == DFM_LITERAL_INT ? float(integer) : float(floating);
      if (typed(r)) {
        uint64_t word = 0;
        memcpy(&word, &value, 4);
        memcpy(static_cast<char *>(literal.contents) + i * 8, &word, 8);
      } else
        memcpy(static_cast<char *>(literal.contents) + i * 4, &value, 4);
    } else {
      int64_t value = op == DFM_LITERAL_BOOL ? int64_t(floating) : integer;
      auto bytes = width(dtype);
      uint64_t bits = uint64_t(value);
      if (op == DFM_LITERAL_INT && bytes < 8) {
        if (unsignedType(dtype)) {
          if (bits > ((uint64_t(1) << (bytes * 8)) - 1))
            fail("Metal unsigned literal overflow");
        } else {
          int64_t hi = (int64_t(1) << (bytes * 8 - 1)) - 1;
          if (value < -hi - 1 || value > hi)
            fail("Metal signed literal overflow");
        }
      }
      memcpy(static_cast<char *>(literal.contents) + i * storageWidth(r), &bits,
             typed(r) ? 8 : bytes);
    }
  }
  auto *descriptors = static_cast<GPUOut *>(out.contents);
  for (int64_t i = 0; i < r.output_count; ++i)
    descriptors[i] = {layout.physical[r.outputs[i].slot],
                      r.outputs[i].reduction, r.outputs[i].min_count,
                      r.outputs[i].dtype};
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
    dispatch(typed(r) ? "unpack" + std::to_string(i) : "unpack", shape.cap);
    auto &input = r.inputs[i];
    offset += ((r.rows + input.bit_offset + 7) / 8) *
              ((input.dtype == DFM_BOOL) + input.has_validity);
  }
  for (auto i : layout.ends) {
    auto &step = r.steps[i];
    if (step.filter == 2) {
      auto orderA = rank, orderB = blocks;
      buffer(0, orderA);
      buffer(1, meta);
      dispatch("order_init", shape.cap);
      for (int64_t stride = 1; stride < shape.cap; stride *= 2) {
        uint32_t span = stride;
        buffer(0, x);
        buffer(1, v);
        buffer(2, orderA);
        buffer(3, orderB);
        buffer(4, meta);
        bytes(5, &span, 4);
        dispatch("order_merge" + std::to_string(i), shape.cap);
        std::swap(orderA, orderB);
      }
      for (int64_t j = 0; j < step.gather_count; ++j) {
        auto physical = layout.physical[r.gathers[step.gather_start + j]];
        if (physical < 0)
          continue;
        uint64_t slot = physical;
        buffer(0, x);
        buffer(1, v);
        buffer(2, y);
        buffer(3, w);
        buffer(4, orderA);
        buffer(5, meta);
        bytes(6, &slot, 8);
        dispatch("order_gather", shape.cap);
      }
      std::swap(x, y);
      std::swap(v, w);
      continue;
    }
    buffer(0, x);
    buffer(1, v);
    buffer(2, literal);
    buffer(3, meta);
    buffer(4, arithmeticError);
    dispatch("project" + std::to_string(i), shape.cap);
    if (step.filter) {
      uint64_t slot = layout.physical[step.slot];
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
        auto physical = layout.physical[r.gathers[step.gather_start + j]];
        if (physical < 0)
          continue;
        uint64_t params[2] = {slot, static_cast<uint64_t>(physical)};
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
    if (fault & 0x20000000u)
      fail("Metal unsupported [precision]: integer to Float32 double-rounding "
           "boundary requires CPU execution");
    if (fault & 0x40000000u)
      fail("Metal strict cast failed: value out of target range or NaN has no "
           "Boolean value");
    if (fault & 0x80000000u)
      fail("Metal unsupported [precision]: Float32 subnormal arithmetic or "
           "underflow requires CPU execution");
    if (fault & 0x10000000U)
      fail("Integer power requires a nonnegative exponent");
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
          ((r.outputs[i].dtype == DFM_INT32 &&
            (values[i].total < INT32_MIN || values[i].total > INT32_MAX)) ||
           (r.outputs[i].dtype == DFM_UINT32 &&
            (values[i].total < 0 || uint64_t(values[i].total) > UINT32_MAX))))
        fail("Integer overflow in Metal sum");
  }
  for (int64_t i = 0; i < r.output_count; ++i) {
    auto &o = r.outputs[i];
    if (r.reductions) {
      if (!rows)
        continue;
      auto a = static_cast<Partial *>(result.contents)[i];
      bool valid = (o.reduction == DFM_MIN || o.reduction == DFM_MAX)
                       ? a.count != 0
                       : o.reduction != DFM_SUM || a.count >= o.min_count;
      if (!valid)
        a.total = 0;
      memcpy(o.values, &a.total, width(o.dtype));
      o.validity[0] = valid;
      stats.download_bytes += width(o.dtype) + 1;
    } else {
      int64_t packed = (rows + 7) / 8;
      auto *base =
          static_cast<uint8_t *>(bits.contents) + (2 * i) * shape.packed;
      if (packed)
        memcpy(o.validity, base, packed);
      int64_t size = o.dtype == DFM_BOOL ? packed : rows * width(o.dtype);
      if (size && typed(r) && o.dtype != DFM_BOOL) {
        auto *words = static_cast<uint64_t *>(x.contents) +
                      layout.physical[o.slot] * shape.cap;
        for (int64_t row = 0; row < rows; ++row)
          memcpy(static_cast<char *>(o.values) + row * width(o.dtype),
                 words + row, width(o.dtype));
      } else if (size)
        memcpy(o.values,
               o.dtype == DFM_BOOL
                   ? base + shape.packed
                   : static_cast<uint8_t *>(x.contents) +
                         layout.physical[o.slot] * shape.cap * storageWidth(r),
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
