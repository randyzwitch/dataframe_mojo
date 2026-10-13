#include "dfmetal.h"
#include "dfmetal_float64.h"
#include "dfmetal_decimal.h"
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
#include <sys/sysctl.h>
#include <unistd.h>

#ifndef DFM_BUILD_ID
#define DFM_BUILD_ID "development"
#endif
static_assert(sizeof(DFMInput) == 40 && sizeof(DFMStep) == 48);
static_assert(sizeof(DFMOutput) == 48 && sizeof(DFMRequest) == 176);
static_assert(sizeof(DFMMemory) == 64 && sizeof(DFMStats) == 136);
static_assert(sizeof(DFMDeviceInfo) == 48);
static_assert(sizeof(DFMWorkspaceInfo) == 40);
static_assert(sizeof(DFMProbeInfo) == 64);
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
  return op == DFM_ADD || op == DFM_SUB || op == DFM_MUL || op == DFM_DIV || op == DFM_GT ||
         op == DFM_EQ ||
         ((op >= DFM_LT && op <= DFM_NE) ||
          (op >= DFM_FLOORDIV && op <= DFM_CLIP_HIGH)) ||
         (op >= DFM_AND && op <= DFM_KEEP_NULLS);
}
bool unaryOp(int64_t op) {
  return (op >= DFM_DT_YEAR && op <= DFM_DT_TRUNCATE) || op == DFM_INT_IS_IN || op == DFM_CAST || op == DFM_NEG || op == DFM_ABS || op == DFM_FLOOR ||
         op == DFM_CEIL || op == DFM_ROUND || op == DFM_SQRT ||
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
  if (t == DFM_DECIMAL128) return 16;
  if (t == DFM_FLOAT32 || t == DFM_INT32 || t == DFM_UINT32)
    return 4;
  if (t == DFM_INT64 || t == DFM_UINT64 || t == DFM_FLOAT64_STORAGE)
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
  return t != DFM_BOOL;
}
int64_t sumType(int64_t t) {
  if (t == DFM_FLOAT32 || t == DFM_FLOAT64_STORAGE) return t;
  return t == DFM_INT32 || t == DFM_UINT32 || t == DFM_UINT64 ? t : DFM_INT64;
}
int64_t reductionType(int64_t type, int64_t op) {
  if (op == DFM_WIDE_INTEGER_SUM || op == DFM_WIDE_INTEGER_MEAN)
    return type == DFM_INT64 || type == DFM_UINT64
               ? (op == DFM_WIDE_INTEGER_SUM ? type : DFM_FLOAT64_STORAGE) : -1;
  if (op == DFM_WIDE_DECIMAL_MEAN)
    return type == DFM_INT32 || type == DFM_INT64 ? DFM_FLOAT64_STORAGE : -1;
  if (op == DFM_WIDE_SUM)
    return type == DFM_INT32 || type == DFM_INT64 ? DFM_DECIMAL128 : -1;
  if (type == DFM_DECIMAL128) {
    if (op == DFM_COUNT || op == DFM_LEN || op == DFM_NULL_COUNT) return DFM_INT64;
    if (op == DFM_MIN || op == DFM_MAX || op == DFM_FIRST || op == DFM_LAST) return DFM_DECIMAL128;
    return -1;
  }
  if (op == DFM_MEAN) return type == DFM_BOOL ? -1 : DFM_FLOAT64_STORAGE;
  if (op == DFM_SUM)
    return sumSupported(type) ? sumType(type) : -1;
  if (op == DFM_COUNT || op == DFM_LEN || op == DFM_NULL_COUNT)
    return DFM_INT64;
  if (op == DFM_ARG_MIN || op == DFM_ARG_MAX)
    return DFM_UINT32;
  if (op == DFM_ANY || op == DFM_ALL)
    return type == DFM_BOOL ? DFM_BOOL : -1;
  if (op == DFM_MIN || op == DFM_MAX || op == DFM_FIRST || op == DFM_LAST)
    return type;
  return -1;
}
bool typed(const DFMRequest &r) { return r.dtype == 0; }
int64_t partialWidth(const DFMRequest &r) {
  if (!r.reductions) return 16;
  for (int64_t i = 0; i < r.output_count; ++i) {
    auto &o = r.outputs[i];
    if (o.dtype == DFM_DECIMAL128 || o.reduction == DFM_WIDE_INTEGER_SUM ||
        o.reduction == DFM_WIDE_INTEGER_MEAN ||
        o.reduction == DFM_WIDE_DECIMAL_MEAN) return 32;
  }
  return 16;
}
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
  case DFM_DECIMAL128:
    return "ulong2";
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
  case DFM_FLOAT64_STORAGE:
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
#include "dfmetal_order.inc"
// A segment ends at a stable filter boundary. Values produced and consumed
// inside one segment stay in per-thread variables; only boundary-live slots
// occupy shared memory. Logical ABI slot identifiers remain unchanged.
struct Layout {
  std::array<int64_t, DFM_MAX_SLOTS> physical;
  std::array<std::array<bool, DFM_MAX_SLOTS>, DFM_MAX_STEPS> gathered{},
      group_after{};
  std::vector<int64_t> ends;
  int64_t slots = 0;
  explicit Layout(const DFMRequest &r) {
    physical.fill(-1);
    auto keep = [&](int64_t slot) {
      if (physical[slot] < 0)
        { physical[slot] = slots;
          slots += slotType(r, slot) == DFM_DECIMAL128 ? 2 : 1; }
    };
    for (int64_t i = 0; i < r.input_count; ++i)
      keep(i);
    for (int64_t first = 0; first < r.step_count;) {
      if (r.steps[first].filter >= 2) {
        ends.push_back(first);
        auto &step = r.steps[first];
        if (step.filter == 4) {
          keep(step.slot);
          for (int64_t j = 0; j < step.nodes; ++j)
            keep(r.code[4 * (step.start + j) + 1]);
        } else if (step.filter == 2) {
          for (int64_t j = 0; j < step.nodes; ++j)
            keep(r.code[4 * (step.start + j) + 1]);
        } else {
          keep(step.slot);
          auto header = r.code + 4 * step.start;
          keep(header[2]);
          keep(header[3]);
          for (int64_t j = 1; j < step.nodes; ++j) {
            auto c = r.code + 4 * (step.start + j);
            keep(c[1]);
            if (c[0] == DFM_GROUP_OUTPUT)
              keep(c[2]);
          }
        }
        ++first;
        continue;
      }
      int64_t end = first;
      while (end + 1 < r.step_count && !r.steps[end].filter &&
             r.steps[end + 1].filter < 2)
        ++end;
      ends.push_back(end);
      std::array<bool, DFM_MAX_SLOTS> local{};
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
    // A column can need physical storage before a boundary without needing a
    // gather after it. Walk uses backward to distinguish those lifetimes.
    std::array<bool, DFM_MAX_SLOTS> live{};
    for (int64_t i = 0; i < r.output_count; ++i)
      live[r.outputs[i].slot] = true;
    for (int64_t i = r.step_count; i-- > 0;) {
      auto &step = r.steps[i];
      if (step.filter == 1 || step.filter == 2)
        for (int64_t j = 0; j < step.gather_count; ++j) {
          auto slot = r.gathers[step.gather_start + j];
          gathered[i][slot] = live[slot];
        }
      if (step.filter == 3) {
        group_after[i] = live;
        auto header = r.code + 4 * step.start;
        live[step.slot] = live[header[2]] = live[header[3]] = false;
        for (int64_t j = 1; j < step.nodes; ++j) {
          auto node = r.code + 4 * (step.start + j);
          if (node[0] == DFM_GROUP_OUTPUT)
            live[node[2]] = false;
        }
        for (int64_t j = 1; j < step.nodes; ++j)
          live[r.code[4 * (step.start + j) + 1]] = true;
      } else if (step.filter == 2 || step.filter == 4) {
        if (step.filter == 4) live[step.slot] = false;
        for (int64_t j = 0; j < step.nodes; ++j)
          live[r.code[4 * (step.start + j) + 1]] = true;
      } else {
        live[step.slot] = false;
        // Retain inputs even for a dead result: evaluating a checked expression
        // can still raise an observable error at this stage.
        for (int64_t j = 0; j < step.nodes; ++j) {
          auto node = r.code + 4 * (step.start + j);
          if (node[0] == DFM_COLUMN)
            live[node[3]] = true;
        }
      }
    }
  }
};
#include "dfmetal_rank.inc"
struct Shape {
  int64_t cap, packed, blocks, bitmap, matrix, validity, literal, outputBits,
      partial;
  bool filter, order, grouped, radixSort, hashGroups;
};
bool hashGroupCandidate(const DFMRequest &r, int64_t step) {
  return typed(r) && r.steps[step].filter == 3 && hashCandidate(r.rows);
}
int64_t hashGroupPasses(const DFMRequest &r) {
  int64_t bits = 0;
  for (auto capacity = hashCapacity(r.rows); capacity > 1; capacity >>= 1) ++bits;
  return (bits + 3) / 4;
}
int64_t radixPasses(const DFMRequest &r, int64_t step) {
  if (!typed(r) || r.steps[step].filter != 2 || r.rows < 65536 || r.rows > 8388608)
    return 0;
  for (int64_t i = 0; i < step; ++i)
    if (r.steps[i].filter == 3) return 0;
  int64_t passes = 0;
  const auto &v = r.steps[step];
  for (int64_t j = 0; j < v.nodes; ++j) {
    auto type = slotType(r, r.code[4 * (v.start + j) + 1]);
    if (type == DFM_FLOAT32 || type == DFM_FLOAT64_STORAGE) return 0;
    passes += 2 * width(type) + 1;
  }
  return passes <= 18 ? passes : 0;
}
DFMMemory estimate(const DFMRequest &, const Shape &, const DFMResidentInput *, bool, bool);
bool directFilteredReduction(const DFMRequest &r) {
  if (!r.reductions || !r.steps || !r.outputs)
    return false;
  bool filter = false;
  for (int64_t i = 0; i < r.step_count; ++i) {
    if (r.steps[i].filter >= 2)
      return false;
    for (int64_t j = 0; j < r.steps[i].nodes; ++j)
      if (r.code[4 * (r.steps[i].start + j)] == DFM_SLICE)
        return false;
    filter |= r.steps[i].filter == 1;
  }
  for (int64_t i = 0; i < r.output_count; ++i)
    if (r.outputs[i].reduction != DFM_SUM &&
        r.outputs[i].reduction != DFM_COUNT &&
        r.outputs[i].reduction != DFM_LEN)
      return false;
  return filter;
}
Shape validate(const DFMRequest &r, bool storage = false,
               const DFMResidentInput *resident = nullptr,
               bool resident_output = false) {
  if (r.abi_version != DFM_ABI_VERSION)
    fail("Metal ABI version mismatch");
  if (r.rows < 0 || r.rows > INT32_MAX || r.slots < 1 || r.slots > DFM_MAX_SLOTS ||
      r.input_count < 1 || r.input_count > r.slots || r.step_count < 0 ||
      r.step_count > DFM_MAX_STEPS || r.output_count < 1 || r.output_count > DFM_MAX_SLOTS ||
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
  if (r.dtype == DFM_DECIMAL128)
    fail("Metal Decimal128 requires typed storage");
  if (r.dtype == DFM_BOOL)
    fail("Metal common storage must be numeric");
  if (r.dtype == DFM_FLOAT64_STORAGE)
    fail("Metal opaque Float64 requires typed storage");
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
    bool native = resident && resident[i].table;
    if (storage && !native && r.rows &&
        (!v.values || (v.has_validity && !v.validity)))
      fail("Missing Metal input data");
    int64_t bytes = (r.rows + v.bit_offset + 7) / 8;
    if (native)
      continue;
    if (v.dtype == DFM_BOOL)
      s.bitmap = add(s.bitmap, bytes);
    else if (typed(r))
      s.bitmap = add(aligned(s.bitmap), mul(r.rows, width(v.dtype)));
    if (v.has_validity)
      s.bitmap = add(s.bitmap, bytes);
  }
  std::array<bool, DFM_MAX_SLOTS> defined{};
  for (int64_t i = 0; i < r.input_count; ++i)
    defined[i] = true;
  for (int64_t i = 0; i < r.step_count; ++i) {
    const auto &v = r.steps[i];
    if (v.start < 0 || v.nodes < 1 || v.nodes > DFM_MAX_NODES ||
        add(v.start, v.nodes) > r.literal_count ||
        (v.filter != 2 && v.slot < r.input_count) || v.slot < 0 ||
        v.slot >= r.slots || v.gather_start < 0 || v.gather_count < 0 ||
        add(v.gather_start, v.gather_count) > r.gather_count ||
        (v.filter < 0 || v.filter > 4))
      fail("Invalid Metal expression step");
    if (v.filter == 4) {
      auto h = r.code + 4 * v.start;
      if (!typed(r) || v.gather_count || defined[v.slot] || h[0] != DFM_RANK_KEY ||
          h[2] < 0 || h[2] > 1 || h[3] < 0 || h[3] > 4 ||
          slotType(r, v.slot) != (h[3] == 0 ? DFM_FLOAT64_STORAGE : DFM_INT64))
        fail("Invalid Metal rank descriptor");
      for (int64_t j = 0; j < v.nodes; ++j) {
        auto key = r.code + 4 * (v.start + j);
        if (key[1] < 0 || key[1] >= r.slots || !defined[key[1]] ||
            r.node_types[v.start + j] != slotType(r, key[1]) ||
            (j && (key[0] != DFM_SORT_KEY || key[2] != 0 || key[3] != 1)))
          fail("Invalid Metal rank input or partition key");
      }
      defined[v.slot] = true;
      s.order = true;
      continue;
    }
    if (v.filter == 3) {
      if (!typed(r) || v.gather_count || v.nodes < 1)
        fail("Invalid Metal group storage");
      auto h = r.code + 4 * v.start;
      if ((h[0] != DFM_GROUP_HEADER && h[0] != DFM_GROUP_HEADER_LAST &&
           h[0] != DFM_GROUP_HEADER_WINDOW) ||
          h[1] < 0 || h[1] >= v.nodes ||
          h[2] < r.input_count || h[2] >= r.slots || h[3] < r.input_count ||
          h[3] >= r.slots || h[2] == h[3] || h[2] == v.slot || h[3] == v.slot)
        fail("Invalid Metal group header");
      if (h[0] == DFM_GROUP_HEADER_WINDOW &&
          (r.literals[v.start] < 1 || r.literals[v.start] > uint64_t(INT32_MAX)))
        fail("Invalid Metal window worker metadata");
      if (defined[v.slot] || defined[h[2]] || defined[h[3]] ||
          slotType(r, v.slot) != DFM_INT64 || slotType(r, h[2]) != DFM_INT64 ||
          slotType(r, h[3]) != DFM_INT64)
        fail("Invalid Metal group temporary slots");
      std::array<bool, DFM_MAX_SLOTS> targets{};
      for (int64_t j = 1; j < v.nodes; ++j) {
        auto c = r.code + 4 * (v.start + j);
        if (c[1] < 0 || c[1] >= r.slots || !defined[c[1]])
          fail("Metal group references an undefined slot");
        if (j <= h[1]) {
          if (c[0] != DFM_GROUP_KEY || c[2] || c[3] ||
              r.node_types[v.start + j] != slotType(r, c[1]))
            fail("Invalid Metal group key");
        } else {
          if (c[0] != DFM_GROUP_OUTPUT || c[2] < r.input_count ||
              c[2] >= r.slots || defined[c[2]] || targets[c[2]] ||
              c[2] == v.slot || c[2] == h[2] || c[2] == h[3])
            fail("Invalid Metal group output");
          auto t = slotType(r, c[1]), out = slotType(r, c[2]);
          if (out != reductionType(t, c[3]))
            fail("Unsupported Metal grouped reduction precision");
          if (c[3] == DFM_WIDE_DECIMAL_MEAN &&
              r.literals[v.start + j] > uint64_t(t == DFM_INT32 ? 9 : 18))
            fail("Invalid Metal decimal mean scale");
          if ((c[3] == DFM_ANY || c[3] == DFM_ALL) &&
              r.literals[v.start + j] > 1)
            fail("Invalid Metal Boolean reduction metadata");
          if (r.node_types[v.start + j] != out ||
              r.literals[v.start + j] > uint64_t(INT64_MAX))
            fail("Invalid Metal group output metadata");
          targets[c[2]] = true;
        }
      }
      for (int64_t j = 0; j < r.slots; ++j)
        defined[j] |= targets[j];
      defined[v.slot] = defined[h[2]] = defined[h[3]] = true;
      s.order = s.grouped = true;
      continue;
    }
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
                  c[0] == DFM_LITERAL_NULL || c[0] == DFM_SLICE;
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
      if (c[0] == DFM_SLICE) {
        if (!typed(r) || nodeType(r, v.start + j) != DFM_BOOL ||
            c[1] < 0 || c[2] < 0 || c[1] >= r.literal_count || c[2] >= r.literal_count || c[3] != -1)
          fail("Invalid Metal slice descriptor");
        for (auto parameter : {c[1], c[2]})
          if (r.code[4 * parameter] != DFM_LITERAL_INT || nodeType(r, parameter) != DFM_INT64)
            fail("Invalid Metal slice parameter dtype");
        if (int64_t(r.literals[c[2]]) < -1)
          fail("Slice length must be nonnegative");
      }
      auto operand = (binary || unary || conditional)
                         ? nodeType(r, v.start + c[1])
                         : nodeType(r, v.start + j);
      auto resultType = nodeType(r, v.start + j);
      if (c[0] == DFM_LITERAL_INT && resultType == DFM_DECIMAL128) {
        if (c[3] < 0 || c[3] >= r.literal_count ||
            r.code[4*c[3]] != DFM_LITERAL_INT || nodeType(r,c[3]) != DFM_INT64)
          fail("Invalid Metal Decimal128 literal high-word payload");
      }
      if ((operand == DFM_DECIMAL128 || resultType == DFM_DECIMAL128) &&
          c[0] != DFM_COLUMN && c[0] != DFM_LITERAL_NULL && c[0] != DFM_LITERAL_INT &&
          c[0] != DFM_GT && c[0] != DFM_LT && c[0] != DFM_GE && c[0] != DFM_LE && c[0] != DFM_EQ && c[0] != DFM_NE &&
          c[0] != DFM_IS_NULL && c[0] != DFM_IS_NOT_NULL &&
          c[0] != DFM_FILL_NULL && c[0] != DFM_KEEP_NULLS && c[0] != DFM_WHEN)
        fail("Metal Decimal128 row arithmetic requires CPU execution");
      if (c[0] >= DFM_DT_YEAR && c[0] <= DFM_DT_TRUNCATE) {
        if (!typed(r) || operand != DFM_INT64 || resultType != DFM_INT64 ||
            (c[0] == DFM_DT_TRUNCATE ? c[2] <= 0 : c[2] != -1) || c[3] < 0 || c[3] > 4 ||
            (c[3] == 4 && c[0] <= DFM_DT_DAY) ||
            (c[3] == 0 && c[0] >= DFM_DT_HOUR && c[0] <= DFM_DT_NANOSECOND) ||
            ((c[0] == DFM_DT_DATE || c[0] == DFM_DT_TIME) && (c[3] == 0 || c[3] == 4)))
          fail("Invalid Metal temporal field descriptor");
      }
      if ((operand == DFM_FLOAT64_STORAGE || resultType == DFM_FLOAT64_STORAGE) &&
          c[0] != DFM_COLUMN && c[0] != DFM_LITERAL_FLOAT && c[0] != DFM_LITERAL_NULL &&
          c[0] != DFM_IS_NULL && c[0] != DFM_IS_NOT_NULL && c[0] != DFM_FILL_NULL &&
          c[0] != DFM_KEEP_NULLS && c[0] != DFM_WHEN &&
          c[0] != DFM_ADD && c[0] != DFM_SUB && c[0] != DFM_MUL &&
          c[0] != DFM_DIV && c[0] != DFM_FLOORDIV && c[0] != DFM_MOD && c[0] != DFM_CAST &&
          c[0] != DFM_FLOOR && c[0] != DFM_CEIL && c[0] != DFM_ROUND && c[0] != DFM_SQRT &&
          c[0] != DFM_NEG && c[0] != DFM_ABS &&
          c[0] != DFM_GT && c[0] != DFM_LT && c[0] != DFM_GE &&
          c[0] != DFM_LE && c[0] != DFM_EQ && c[0] != DFM_NE &&
          !(c[0] >= DFM_IS_NAN && c[0] <= DFM_IS_INFINITE) &&
          c[0] != DFM_FILL_NAN && c[0] != DFM_CLIP_LOW && c[0] != DFM_CLIP_HIGH)
        fail("Metal Float64 requires CPU execution for this operation");
      if (c[0] == DFM_SQRT &&
          (operand == DFM_BOOL || resultType !=
              (operand == DFM_FLOAT32 ? DFM_FLOAT32 : DFM_FLOAT64_STORAGE)))
        fail("Invalid Metal square root dtype");
      if (c[0] == DFM_INT_IS_IN) {
        if (!typed(r) || operand == DFM_FLOAT32 || operand == DFM_BOOL ||
            nodeType(r, v.start + j) != DFM_BOOL || c[2] < 0 || c[3] < 0 ||
            add(c[2], c[3]) > r.literal_count)
          fail("Invalid Metal integer membership descriptor");
        for (int64_t value = c[2]; value < c[2] + c[3]; ++value)
          if (r.code[4 * value] != DFM_LITERAL_INT || nodeType(r, value) != operand)
            fail("Metal integer membership literal dtype mismatch");
      }
      if ((c[0] >= DFM_IS_NAN && c[0] <= DFM_IS_INFINITE) &&
          operand != DFM_FLOAT32 && operand != DFM_FLOAT64_STORAGE)
        fail("Metal floating classification requires Float32");
      if (c[0] == DFM_POW &&
          operand == DFM_FLOAT32)
        fail("Metal floating operator requires CPU execution");
      if (c[0] == DFM_ROUND && c[3] != 0 &&
          (operand == DFM_FLOAT32 || operand == DFM_FLOAT64_STORAGE) && !typed(r))
        fail("Metal floating decimal rounding requires typed descriptors");
      if (c[0] == DFM_ROUND && c[3] != 0 &&
          (operand == DFM_FLOAT32 || operand == DFM_FLOAT64_STORAGE) &&
          r.literals[v.start+j] > 0x7ff0000000000000UL)
        fail("Invalid Metal floating rounding scale");
      if (nodeType(r, v.start + j) != DFM_FLOAT32 &&
          nodeType(r, v.start + j) != DFM_FLOAT64_STORAGE && c[0] == DFM_LITERAL_FLOAT)
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
        if (c[0] == DFM_FILL_NAN && t != DFM_FLOAT32 && t != DFM_FLOAT64_STORAGE)
          fail("Metal fill NaN requires Float32");
        if ((compare || logic || nulltest) && t != DFM_BOOL)
          fail("Metal predicate dtype mismatch");
        if (logic &&
            (r.node_types[v.start + c[1]] != DFM_BOOL ||
             (c[0] != DFM_NOT && r.node_types[v.start + c[2]] != DFM_BOOL)))
          fail("Metal Boolean operand dtype mismatch");
        if (c[0] == DFM_DIV &&
            (t != DFM_FLOAT64_STORAGE || r.node_types[v.start+c[1]] == DFM_BOOL ||
             r.node_types[v.start+c[1]] != r.node_types[v.start+c[2]]))
          fail("Invalid Metal floating division dtype");
        bool wideCompare = compare && (r.node_types[v.start+c[1]] == DFM_DECIMAL128 || r.node_types[v.start+c[2]] == DFM_DECIMAL128);
        if (wideCompare) {
          for (auto child : {c[1], c[2]}) {
            auto type = r.node_types[v.start+child];
            if (type != DFM_DECIMAL128 && type != DFM_INT32 && type != DFM_INT64)
              fail("Invalid Metal Decimal128 comparison operand");
          }
        }
        if ((compare || math) && !wideCompare &&
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
      auto expected = reductionType(slotType(r, v.slot), v.reduction);
      if (expected < 0)
        fail("Metal reduction requires unsupported accumulator precision");
      if (v.dtype != expected)
        fail("Invalid Metal reduction dtype");
      if (v.reduction == DFM_WIDE_DECIMAL_MEAN &&
          (v.min_count < 0 ||
           v.min_count > (slotType(r, v.slot) == DFM_INT32 ? 9 : 18)))
        fail("Invalid Metal decimal mean scale");
      if (v.reduction != DFM_SUM && v.reduction != DFM_COUNT &&
          v.reduction != DFM_LEN && !typed(r))
        fail("Metal reduction requires typed storage");
      if ((v.reduction == DFM_ANY || v.reduction == DFM_ALL) && v.min_count > 1)
        fail("Invalid Metal Boolean reduction metadata");
    }
    if (storage && !resident_output && (r.rows || r.reductions || s.grouped) &&
        (!v.values || !v.validity))
      fail("Missing Metal output storage");
  }
  auto layout = Layout(r);
  s.matrix = mul(mul(s.cap, layout.slots), storageWidth(r));
  s.validity = mul(s.cap, layout.slots);
  s.literal = std::max<int64_t>(1, mul(r.literal_count, storageWidth(r)));
  s.outputBits = mul(mul(s.packed, r.output_count), 2);
  s.partial =
      r.reductions
          ? mul(mul(std::min<int64_t>(s.blocks, 1024), r.output_count), partialWidth(r))
          : 1;
  if (directFilteredReduction(r))
    s.filter = false;
  for (int64_t i = 0; i < r.step_count; ++i) {
    s.radixSort |= radixPasses(r, i) != 0;
    s.hashGroups |= hashGroupCandidate(r, i);
  }
  if ((s.radixSort || s.hashGroups) && r.memory_budget_bytes >= 0) {
    auto bound = s;
    bound.bitmap = 0;
    for (int64_t i = 0; i < r.input_count; ++i) {
      auto &v = r.inputs[i];
      auto bytes = (r.rows + v.bit_offset + 7) / 8;
      if (v.dtype == DFM_BOOL) bound.bitmap = add(bound.bitmap, bytes);
      else if (typed(r)) bound.bitmap = add(aligned(bound.bitmap), mul(r.rows, width(v.dtype)));
      if (v.has_validity) bound.bitmap = add(bound.bitmap, bytes);
    }
    if (estimate(r, bound, nullptr, false, false).peak_bytes > r.memory_budget_bytes)
      s.radixSort = s.hashGroups = false;
  }
  return s;
}
DFMMemory estimate(const DFMRequest &r, const Shape &s,
                   const DFMResidentInput *resident = nullptr,
                   bool retain = false, bool final = false) {
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
  if (s.grouped)
    account(16);
  if (s.radixSort || s.hashGroups) {
    RadixShape radix(s.cap);
    account(mul(radix.cap, 8)); account(mul(radix.cap, 4));
    account(radix.histogram); account(64); account(32);
  }
  if (s.hashGroups) account(mul(hashCapacity(s.cap), 4));
  account(s.bitmap);
  account(s.literal);
  account(s.outputBits);
  account(s.partial);
  account(mul(r.output_count, 32));
  account(16);
  account(4);
  if (r.reductions)
    account(mul(r.output_count, partialWidth(r)));
  for (int64_t i = 0; i < r.input_count; ++i) {
    if (resident && resident[i].table)
      continue;
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
    if (retain && !final)
      break;
    auto t = r.outputs[i].dtype;
    m.result_bytes =
        add(m.result_bytes,
            r.reductions
                ? width(t) + 1
                : add(t == DFM_BOOL ? s.packed
                                    : mul(s.grouped ? s.cap : r.rows, width(t)),
                      s.packed));
  }
  m.peak_bytes = add(m.shared_bytes, m.result_bytes);
  auto layout = Layout(r);
  m.launches = r.input_count + layout.ends.size() +
      (r.reductions ? 2 : (!retain || final));
  for (int64_t i = 0; i < r.step_count; ++i)
    if (r.steps[i].filter && !directFilteredReduction(r)) {
      if (r.steps[i].filter == 1)
        m.launches += 2;
      else if (s.hashGroups && hashGroupCandidate(r, i)) {
        m.launches += 3 + 3 * hashGroupPasses(r);
        // Restore the public key order on the reduced table, proportional to G.
        m.launches += 2;
        for (int64_t stride = 1; stride < s.cap; stride *= 2) ++m.launches;
        for (int64_t slot = 0; slot < r.slots; ++slot)
          m.launches += layout.group_after[i][slot];
      }
      else if (s.radixSort && radixPasses(r, i))
        m.launches += 1 + 2 * r.steps[i].nodes + 3 * radixPasses(r, i);
      else
        for (int64_t stride = 1; stride < s.cap; stride *= 2)
          ++m.launches;
      if (r.steps[i].filter == 3)
        m.launches += 5;
      if (r.steps[i].filter == 4)
        m.launches += r.code[4 * r.steps[i].start + 3] == 3 ? 3 : 1;
      for (int64_t j = 0; j < r.steps[i].gather_count; ++j)
        m.launches += layout.gathered[i][r.gathers[r.steps[i].gather_start + j]];
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
  // Idle buffers have no in-flight users. The independent lock also covers
  // join-session destruction, which happens outside the submission lock.
  std::mutex workspace_mutex;
  std::vector<id<MTLBuffer>> idle;
  int64_t idle_bytes = 0, workspace_limit = 0;
  int64_t workspace_hits = 0, workspace_allocations = 0;
  int64_t headroom() {
    std::lock_guard<std::mutex> lock(workspace_mutex);
    auto recommended = int64_t(std::min<uint64_t>(
        device.recommendedMaxWorkingSetSize, INT64_MAX));
    auto allocated = int64_t(std::min<uint64_t>(device.currentAllocatedSize,
                                              INT64_MAX));
    // Idle buffers are reclaimable. Live buffers from other sessions remain
    // charged to allocated; they can never satisfy this request's allocations.
    return std::max<int64_t>(0, recommended -
                                std::max<int64_t>(0, allocated - idle_bytes));
  }
  id<MTLBuffer> acquire(int64_t bytes) {
    std::lock_guard<std::mutex> lock(workspace_mutex);
    for (auto it = idle.begin(); it != idle.end(); ++it) {
      if (int64_t((*it).length) == bytes) {
        auto buffer = *it;
        idle.erase(it);
        idle_bytes -= bytes;
        ++workspace_hits;
        // Reset reused storage so inactive lanes and bitmap padding cannot
        // expose data from a previous collection, including failed requests.
        memset(buffer.contents, 0, bytes);
        return buffer;
      }
    }
    // A shape change may need new allocations. Reclaim unmatched idle buffers
    // first, rather than allowing a cache to consume the request's headroom.
    idle.clear();
    idle_bytes = 0;
    auto buffer = [device newBufferWithLength:bytes
                                    options:MTLResourceStorageModeShared |
                                            MTLResourceCPUCacheModeDefaultCache];
    if (!buffer)
      fail("Metal shared buffer allocation failed");
    ++workspace_allocations;
    return buffer;
  }
  void recycle(std::vector<id<MTLBuffer>> &buffers) noexcept {
    std::lock_guard<std::mutex> lock(workspace_mutex);
    for (auto buffer : buffers) {
      auto bytes = int64_t(buffer.length);
      if (idle.size() >= 64 || bytes > workspace_limit - idle_bytes)
        continue;
      try {
        idle.push_back(buffer);
        idle_bytes += bytes;
      } catch (...) {
        // Caching is optional; allocation failure during destruction must not
        // interrupt context release or ownership cleanup.
        break;
      }
    }
    buffers.clear();
  }
};
std::mutex defaultMutex;
Context *defaultContext = nullptr;
std::string residentSource() {
  return R"MSL(
struct ResidentCopy { ulong source, validity, target, rows;
 long scalar, minimum, sum, padding; ulong source_high,target_high; };
kernel void resident_copy(device const ulong *a[[buffer(0)]],
 device const uchar *b[[buffer(1)]],device ulong *x[[buffer(2)]],
 device uchar *v[[buffer(3)]],constant ResidentCopy &p[[buffer(4)]],
 uint row[[thread_position_in_grid]]) {
 if(row>=p.rows)return;
 bool valid=p.scalar? (p.sum? *((device const long*)(b+p.validity))>=p.minimum:
                              *((device const long*)(b+p.validity))>0):
                         b[p.validity+row]!=0;
 x[p.target+row]=valid?a[p.source+row]:0UL;
 v[p.target+row]=uchar(valid);
 if(p.target_high){x[p.target_high+row]=valid?a[p.source_high+row]:0UL;v[p.target_high+row]=uchar(valid);}
}
)MSL";
}
std::string castExpression(int64_t from, int64_t to, const std::string &a,
                           const std::string &z, const std::string &valid,
                           bool strict, int64_t id) {
  auto target = std::string(metalType(to));
  std::string body = "bool fits=true;";
  if (to == DFM_FLOAT64_STORAGE) {
    body += z + "=f64_word(" + encode(from,a) + "," + std::to_string(from) + ");";
  } else if (from == DFM_FLOAT64_STORAGE) {
    if (to == DFM_BOOL)
      body += "fits=!f64_nan(" + a + ");" + z + "=uchar(f64_abs(" + a + ")!=0);";
    else if (to == DFM_FLOAT32)
      body += z + "=as_type<float>(f64_to_float32(" + a + "));";
    else
      body += "F64Integer casted=f64_to_integer(" + a + "," + std::to_string(width(to)*8) +
              "," + (unsignedType(to) ? "true" : "false") + ");fits=casted.valid;if(fits)" + z + "=" + target + "(casted.word);";
  } else if (to == DFM_BOOL) {
    if (from == DFM_FLOAT32)
      body += "uint ab=as_type<uint>(" + a +
              ")&0x7fffffffu;fits=ab<=0x7f800000u;" + z + "=uchar(ab!=0);";
    else
      body += z + "=uchar(" + a + "!=0);";
  } else if (to == DFM_FLOAT32) {
    if (from == DFM_INT64 || from == DFM_UINT64) {
      // CPU conversion rounds through binary64 before narrowing to Float32.
      body += z + "=as_type<float>(f64_to_float32(f64_word(ulong(" + a + ")," + std::to_string(from) + ")));";
    } else
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
std::string orderingSource() {
  return R"MSL(
bool wide_less(ulong2 a,ulong2 b) {
 return a.y==b.y?a.x<b.x:long(a.y)<long(b.y);
}
bool wide_compare(ulong2 a,ulong2 b,long op) {
 bool equal=all(a==b),less=wide_less(a,b);
 if(op==9)return equal;if(op==23)return !equal;
 if(op==20)return less;if(op==8)return !equal&&!less;
 if(op==21)return !less;return equal||less;
}
ulong2 wide_word(long value){return ulong2(ulong(value),value<0?~0UL:0UL);}
bool extrema_better(ulong a,ulong b,long type,bool maximum) {
 if(type==11){ulong ax=a&0x7fffffffffffffffUL,ay=b&0x7fffffffffffffffUL;
 bool nx=ax>0x7ff0000000000000UL,ny=ay>0x7ff0000000000000UL;
 if(nx||ny)return nx!=ny&&(maximum?nx:ny);
 if(ax==0&&ay==0)return false;
 ulong kx=(a&0x8000000000000000UL)?~a:(a^0x8000000000000000UL);
 ulong ky=(b&0x8000000000000000UL)?~b:(b^0x8000000000000000UL);
 return maximum?kx>ky:kx<ky;}
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
)MSL";
}

std::string groupSource(const DFMRequest &r, const Layout &layout) {
  std::string text;
  for (int64_t i = 0; i < r.step_count; ++i) {
    auto &step = r.steps[i];
    if (step.filter != 3)
      continue;
    auto h = r.code + 4 * step.start;
    auto keys = h[1];
    bool lastRepresentative = h[0] == DFM_GROUP_HEADER_LAST;
    bool windowFloating = h[0] == DFM_GROUP_HEADER_WINDOW;
    auto name = std::to_string(i);
    auto index = std::to_string(layout.physical[step.slot]) + "UL*meta.z";
    auto starts = std::to_string(layout.physical[h[2]]) + "UL*meta.z";
    auto first = layout.physical[h[3]];
    text += "bool group_equal" + name +
            "(uint a,uint b,device const T *x,device const uchar *v,constant "
            "uint4 &meta){";
    for (int64_t j = 0; j < keys; ++j) {
      auto c = r.code + 4 * (step.start + 1 + j);
      auto slot = std::to_string(layout.physical[c[1]]) + "UL*meta.z";
      auto type = std::to_string(slotType(r, c[1]));
      if (slotType(r, c[1]) == DFM_DECIMAL128)
        text += "{ulong ai=" + slot + "+a,bi=" + slot + "+b;if(v[ai]!=v[bi])return false;if(v[ai]&&(x[ai]!=x[bi]||x[ai+meta.z]!=x[bi+meta.z]))return false;}";
      else
        text += "{ulong ai=" + slot + "+a,bi=" + slot +
                "+b;bool av=v[ai]!=0,bv=v[bi]!=0;if(av!=bv)return "
                "false;if(av&&(extrema_better(x[ai],x[bi]," +
                type + ",false)||extrema_better(x[bi],x[ai]," + type +
                ",false)))return false;}";
    }
    text += "return true;}\n";
    text +=
        "kernel void group_mark" + name +
        "(device T *x[[buffer(0)]],device uchar *v[[buffer(1)]],device const "
        "uint *index[[buffer(2)]],constant uint4 &meta[[buffer(3)]],uint "
        "row[[thread_position_in_grid]]){if(row>=meta.x)return;x[" +
        index + "+row]=index[row];x[" + starts + "+row]=row==0||!group_equal" +
        name + "(index[row-1],index[row],x,v,meta);v[" + starts + "+row]=1;}\n";
    text +=
        "kernel void group_offsets" + name +
        "(device uint *blocks[[buffer(0)]],device uint4 "
        "&meta[[buffer(1)]]){uint total=0;for(uint "
        "i=0;i<(meta.x+255)/256;++i){uint "
        "count=blocks[i];blocks[i]=total;total+=count;}meta.y=meta.x;meta.x=" +
        std::string(keys ? "total" : "max(1U,total)") + ";}\n";
    text += "kernel void group_starts" + name +
            "(device T *x[[buffer(0)]],device const uchar "
            "*v[[buffer(1)]],device const uint *rank[[buffer(2)]],device const "
            "uint *blocks[[buffer(3)]],constant uint4 "
            "&meta[[buffer(4)]],device uint *args[[buffer(5)]],uint "
            "row[[thread_position_in_grid]]){if(row==0){args[0]=max(1U,meta.x);"
            "args[1]=" +
            std::to_string(step.nodes) + ";args[2]=1;if(meta.y==0)x[" + starts +
            "]=0;}if(row>=meta.y)return;uint a=uint(x[" + index +
            "+row]);if(row==0||!group_equal" + name + "(uint(x[" + index +
            "+row-1]),a,x,v,meta))x[" + starts +
            "+rank[row]+blocks[row/256]]=row;}\n";
    // step.nodes = keys + aggregates + header; header corresponds to first-seen
    // output.
    text +=
        "kernel void group_reduce" + name +
        "(device const T *x[[buffer(0)]],device const uchar "
        "*v[[buffer(1)]],device T *y[[buffer(2)]],device uchar "
        "*w[[buffer(3)]],constant uint4 &meta[[buffer(4)]],device atomic_uint "
        "*error[[buffer(5)]],device const T *lit[[buffer(6)]],uint2 "
        "group[[threadgroup_position_in_grid]],uint "
        "lane[[thread_index_in_threadgroup]]){if(group.x>=meta.x)return;uint "
        "begin=uint(x[" +
        starts + "+group.x]),end=group.x+1<meta.x?uint(x[" + starts +
        "+group.x+1]):meta.y;Out o;long target;switch(group.y){";
    text += "case 0:o={0," + std::string(lastRepresentative ? "81" : "204") +
            ",0,3};target=" + std::to_string(first) + ";break;";
    for (int64_t j = 1; j < step.nodes; ++j) {
      auto c = r.code + 4 * (step.start + j);
      bool key = j <= keys;
      auto target = layout.physical[key ? c[1] : c[2]];
      text += "case " + std::to_string(j) + ":o={" +
              std::to_string(layout.physical[c[1]]) + "," +
              std::to_string(key ? -1 : c[3]) + ",long(lit[" +
              std::to_string(step.start + j) + "])," +
              std::to_string(slotType(r, c[1])) +
              "};target=" + std::to_string(target) + ";break;";
    }
    text +=
        R"MSL(default:return;}ulong to=ulong(target)*meta.z+group.x;
 if(o.kind==-1){if(lane==0){uint row=uint(x[)MSL" +
        index +
        "+" + std::string(lastRepresentative ? "end-1" : "begin") +
        R"MSL(]);ulong from=ulong(o.slot)*meta.z+row;y[to]=x[from];w[to]=v[from];if(o.type==12){y[to+meta.z]=x[from+meta.z];w[to+meta.z]=v[from];}}return;}
)MSL" + (windowFloating ?
        // CPU OVER reductions add rows in source order within each worker,
        // then merge contiguous workers in order. Eager select/with_columns
        // use 8192-row batches, which align those worker boundaries.
        "if((o.kind==10||o.kind==82)&&(o.type==1||o.type==11)){if(lane)return;"
        "ulong workers=max(1UL,min(min(ulong(lit[" + std::to_string(step.start) +
        "]),ulong(meta.y)/65536UL),ulong(meta.y)/max(1UL,4UL*ulong(meta.x))));"
        "ulong step=((ulong(meta.y)+workers-1UL)/workers+8191UL)/8192UL*8192UL;"
        "ulong total=0UL;long count=0;uint at=begin;"
        "for(ulong worker=0;worker<workers;++worker){ulong partial=0UL;"
        "ulong stop=min(ulong(meta.y),(worker+1UL)*step);"
        "while(at<end){uint row=uint(x[" + index +
        "+at]);if(ulong(row)>=stop)break;++at;ulong from=ulong(o.slot)*meta.z+row;"
        "if(v[from]){partial=f64_add(partial,f64_word(x[from],o.type));++count;}}"
        "total=worker?f64_add(total,partial):partial;}"
        "long sum=long(total);state_finish(sum,count,o,end-begin);"
        "bool valid=state_valid(o,count);y[to]=valid?ulong(sum):0UL;w[to]=uchar(valid);return;}\n"
        : "") + R"MSL(
 threadgroup long sums[256];threadgroup long counts[256];threadgroup long highs[256];long sum=0,count=0,high=0;
 bool first=group.y==0;
 for(uint at=begin+lane;at<end;at+=256){uint row=uint(x[)MSL" +
        index + R"MSL(+at]);ulong idx=ulong(o.slot)*meta.z+row;
 if(first)state_add(sum,count,ulong(row),true,long(row),o);
 else {if(v[idx]&&!sum_bound(x[idx],o,end-begin)){atomic_fetch_min_explicit(error,0xE0000000U,memory_order_relaxed);continue;}
 if(wide_accum(o)){if(v[idx]){wide_accumulate(sum,high,x[idx],o);++count;}}
 else if(wide_pick(o))state_add_wide(sum,high,count,ulong2(x[idx],x[idx+meta.z]),v[idx]!=0,long((o.kind==95||o.kind==96)?at-begin:row),o);
 else state_add(sum,count,x[idx],v[idx]!=0,long((o.kind==95||o.kind==96)?at-begin:row),o);}}
 sums[lane]=sum;counts[lane]=count;highs[lane]=high;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint stride=128;stride;stride/=2){if(lane<stride){long a=sums[lane],b=counts[lane],h=highs[lane];if(wide_accum(o)){wide_merge(a,h,sums[lane+stride],highs[lane+stride]);b+=counts[lane+stride];}else if(wide_pick(o))state_merge_wide(a,h,b,sums[lane+stride],highs[lane+stride],counts[lane+stride],o);else state_merge(a,b,sums[lane+stride],counts[lane+stride],o);sums[lane]=a;counts[lane]=b;highs[lane]=h;}threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(lane==0){sum=sums[0];high=highs[0];count=counts[0];state_finish_wide(sum,high,count,o,end-begin);bool valid=state_valid(o,count);
 if(o.kind==210&&valid&&(o.type==10?high!=0:high!=(sum<0?-1L:0L)))atomic_fetch_min_explicit(error,0xF0000000U,memory_order_relaxed);
 if(o.kind==10&&valid&&((o.type==2&&(sum<(-2147483647L-1L)||sum>2147483647L))||(o.type==9&&(sum<0||ulong(sum)>4294967295UL))))atomic_fetch_min_explicit(error,0xF0000000U,memory_order_relaxed);
 y[to]=valid?ulong(sum):0UL;w[to]=uchar(valid);if(wide_result(o)){y[to+meta.z]=valid?ulong(high):0UL;w[to+meta.z]=uchar(valid);}}}
)MSL";
  }
  return text;
}

std::string source(const DFMRequest &r) {
  auto layout = Layout(r);
  bool direct = directFilteredReduction(r);
  int64_t filterSlot = 0;
  if (direct)
    for (int64_t i = 0; i < r.step_count; ++i)
      if (r.steps[i].filter)
        filterSlot = layout.physical[r.steps[i].slot];
  std::string text = "#include <metal_stdlib>\nusing namespace metal;\n#pragma "
                     "clang fp contract(off)\n";
  bool wide = partialWidth(r) == 32;
  if (typed(r))
    for (int64_t i = 0; i < r.slots; ++i) wide |= r.slot_types[i] == DFM_DECIMAL128;
  for (int64_t i = 0; i < r.code_words; i += 4)
    wide |= r.code[i] == DFM_GROUP_OUTPUT &&
            (r.code[i + 3] == DFM_WIDE_INTEGER_SUM ||
             r.code[i + 3] == DFM_WIDE_INTEGER_MEAN ||
             r.code[i + 3] == DFM_WIDE_DECIMAL_MEAN);
  text += "constant bool USE_WIDE=" + std::string(wide ? "true" : "false") + ";\n";
  text += "struct Out { long slot; long kind; long minimum; long type; "
          "};\n";
  if (partialWidth(r) == 32)
    text += "struct Partial { long total, high, count, padding; };\n"
            "Partial make_partial(long total,long high,long count){return {total,high,count,0};}\n"
            "long partial_high(Partial p){return p.high;}\n";
  else
    text += "struct Partial { long total,count; };\n"
            "Partial make_partial(long total,long high,long count){return {total,count};}\n"
            "long partial_high(Partial p){return 0;}\n";
  text += residentSource();
  text += dfm_float64_source;
  text += dfm_decimal_source;
  text += "constant bool DIRECT_FILTERED_REDUCTION=" +
          std::string(direct ? "true" : "false") + ";\n" +
          "constant ulong FILTERED_SLOT=" + std::to_string(filterSlot) + "UL;\n";
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
    std::array<bool, 13> used{};
    for (int64_t i = 0; i < r.literal_count; ++i)
      used[r.node_types[i]] = true;
    for (int64_t t = 1; t <= 10; ++t)
      if (used[t] && t != DFM_BOOL)
        text += "namespace d" + std::to_string(t) + " {\n" +
                arithmeticSource(t) + "}\n";
  } else
    text += arithmeticSource(r.dtype);
  text += "ulong read_word(T value){return " +
          std::string(!typed(r) && r.dtype == DFM_FLOAT32
                          ? "ulong(as_type<uint>(value))"
                          : "ulong(value)") +
          ";}\n";
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
 constant ulong4 &p[[buffer(7)]],uint row[[thread_position_in_grid]]) {
 if(row>=meta.y||!v[p.x*meta.z+row]||x[p.x*meta.z+row]==T(0))return;
 ulong target=rank[row]+blocks[row/256];y[p.y*meta.z+target]=x[p.y*meta.z+row];w[p.y*meta.z+target]=v[p.y*meta.z+row];
 if(p.z){y[(p.y+1)*meta.z+target]=x[(p.y+1)*meta.z+row];w[(p.y+1)*meta.z+target]=v[p.y*meta.z+row];}
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
)MSL";
  text += orderingSource();
  text += R"MSL(
bool candidate_better(ulong a,ulong b,Out o){
 if((o.kind==95||o.kind==96)&&o.type==11){
  bool an=f64_nan(a),bn=f64_nan(b);if(an!=bn)return !an;
 }
 if((o.kind==95||o.kind==96)&&o.type==1){
  bool an=(uint(a)&0x7fffffffU)>0x7f800000U,bn=(uint(b)&0x7fffffffU)>0x7f800000U;
  if(an!=bn)return !an;
 }
 return extrema_better(a,b,o.type,o.kind==81||o.kind==96);
}
// First/last use signed (row+1): negative selects a null row; zero is empty.
// Boolean states combine bits for true, false, and null without counting rows.
// Each positive and negative subset must fit independently. This proves
// every parallel partial and merge exact without a wider accumulator. A
// failed proof is unsupported, even if cancellation could fit the result.
bool sum_bound(ulong word,Out o,ulong rows){
 if((o.kind!=10&&o.kind!=82)||(o.type!=3&&o.type!=10)||!rows)return true;
 bool negative=o.type==3&&long(word)<0;
 ulong maximum=o.type==10?~0UL:(negative?(1UL<<63):((1UL<<63)-1UL));
// Variable 64-bit division here crashes the M1 grouped pipeline compiler.
// A native high multiply checks the identical bound without division.
 ulong magnitude=negative?0UL-word:word;
 return mulhi(magnitude,rows)==0UL&&magnitude*rows<=maximum;
}
// At most Int32.MAX rows of 64-bit inputs bound every sum partial below 2**95.
// Two-word modulo addition is therefore exact in signed128 and precision38.
void wide_merge(thread long &low,thread long &high,long other,long upper){
 ulong before=ulong(low),after=before+ulong(other);
 low=long(after);high=long(ulong(high)+ulong(upper)+ulong(after<before));
}
void wide_add(thread long &low,thread long &high,ulong word){
 wide_merge(low,high,long(word),long(word)<0?-1L:0L);
}
bool wide_accum(Out o){return USE_WIDE&&(o.kind==208||o.kind==210||o.kind==211||o.kind==212);}
void wide_accumulate(thread long &low,thread long &high,ulong word,Out o){
 wide_merge(low,high,long(word),o.type==10?0L:(long(word)<0?-1L:0L));
}
bool wide_pick(Out o){return USE_WIDE&&o.type==12&&(o.kind==80||o.kind==81||o.kind==83||o.kind==84);}
bool wide_result(Out o){return USE_WIDE&&(o.kind==208||(wide_pick(o)&&o.kind!=95&&o.kind!=96));}
void state_merge_wide(thread long &value,thread long &high,thread long &count,long other,long upper,long n,Out o){
 if(o.kind==83||o.kind==84){
  long at=count<0?-count:count,bt=n<0?-n:n;
  if(bt&&(!at||(o.kind==83?bt<at:bt>at))){value=other;high=upper;count=n;}return;
 }
 if(!n)return;ulong2 a=ulong2(ulong(value),ulong(high)),b=ulong2(ulong(other),ulong(upper));
 bool better=(o.kind==81||o.kind==96)?wide_less(a,b):wide_less(b,a);
 if(!count||better||(all(a==b)&&n<count)){value=other;high=upper;count=n;}
}
void state_add_wide(thread long &value,thread long &high,thread long &count,ulong2 word,bool valid,long row,Out o){
 long n=(o.kind==83||o.kind==84)?(valid?row+1:-(row+1)):(valid?row+1:0);
 state_merge_wide(value,high,count,valid?long(word.x):0,valid?long(word.y):0,n,o);
}
void state_merge(thread long &value,thread long &count,long other,long n,Out o){
 if(o.kind==83||o.kind==84){
  long at=count<0?-count:count,bt=n<0?-n:n;
  if(bt&&(!at||(o.kind==83?bt<at:bt>at))){value=other;count=n;}return;
 }
 if(o.kind==80||o.kind==81||o.kind==95||o.kind==96||o.kind==204){
  if(!n)return;Out comparison=o;if(o.kind==204)comparison.kind=80;
  bool take=!count||candidate_better(ulong(other),ulong(value),comparison);
  if(!take&&!candidate_better(ulong(value),ulong(other),comparison)&&n<count)take=true;
  if(take){value=other;count=n;}return;
 }
 if(o.kind==91||o.kind==92){value|=other;return;}
 if((o.kind==82||o.kind==10)&&(o.type==1||o.type==11)){
  if(n){value=long(f64_add(ulong(value),ulong(other)));count+=n;}return;
 }
 value=long(ulong(value)+ulong(other));count+=n;
}
void state_add(thread long &value,thread long &count,ulong word,bool valid,long row,Out o){
 if(o.kind==83||o.kind==84){state_merge(value,count,valid?long(word):0,valid?row+1:-(row+1),o);return;}
 if(o.kind==91||o.kind==92){value|=valid?(word?1L:2L):4L;return;}
 if(o.kind==93){count+=!valid;return;}
 if(!valid)return;
 if(o.kind==80||o.kind==81||o.kind==95||o.kind==96||o.kind==204){state_merge(value,count,long(word),row+1,o);return;}
 ++count;
 if((o.kind==82||o.kind==10)&&(o.type==1||o.type==11))value=long(f64_add(ulong(value),f64_word(word,o.type)));
 else if(o.kind==10||o.kind==82)value=long(ulong(value)+word);
}
void state_finish(thread long &value,thread long &count,Out o,long rows){
 if(o.kind==82){if(count){ulong total=(o.type==1||o.type==11)?ulong(value):f64_word(ulong(value),o.type);
  value=long(f64_div(total,f64_from_signed(count)));}}
 else if(o.kind==10&&o.type==1){value=long(ulong(f64_to_float32(ulong(value))));}
 else if(o.kind==11||o.kind==93){value=count;count=1;}
 else if(o.kind==90){value=rows;count=1;}
 else if(o.kind==83||o.kind==84){count=count>0?1:0;}
 else if(o.kind==95||o.kind==96){value=count?count-1:0;count=count?1:0;}
 else if(o.kind==91||o.kind==92){
  bool yes=(value&1)!=0,no=(value&2)!=0,unknown=(value&4)!=0;
  count=o.minimum||!unknown||(o.kind==91?yes:no);value=o.kind==91?yes:!no;
 }
}
void state_finish_wide(thread long &value,thread long &high,thread long &count,Out o,long rows){
 if(o.kind==211){if(count)value=long(f64_div(f64_from_wide(ulong(value),ulong(high),o.type==10),f64_from_signed(count)));high=0;}
 else if(o.kind==212){if(count)value=long(f64_decimal_mean(ulong(value),ulong(high),ulong(count),uint(o.minimum)));high=0;}
 else state_finish(value,count,o,rows);
}
bool state_valid(Out o,long count){return o.kind==10||o.kind==208||o.kind==210?count>=o.minimum:count>0;}
kernel void reduce_rows(device const T *x[[buffer(0)]],device const uchar *v[[buffer(1)]],
 device Partial *partials[[buffer(2)]],device const Out *out[[buffer(3)]],
 constant uint4 &meta[[buffer(4)]],constant ulong &groups[[buffer(5)]],device atomic_uint *error[[buffer(6)]],
 uint2 group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_threadgroup]]) {
 threadgroup long sums[256];threadgroup long counts[256];threadgroup long highs[256];long sum=0,count=0,high=0;
 Out o=out[group.y];
 for(ulong row=group.x*256+lane;row<meta.x;row+=groups*256){ulong idx=o.slot*meta.z+row;
 if(DIRECT_FILTERED_REDUCTION&&(!v[FILTERED_SLOT*meta.z+row]||x[FILTERED_SLOT*meta.z+row]==T(0)))continue;
 Out current=o;if(DIRECT_FILTERED_REDUCTION&&o.kind==90)current.kind=11;
 ulong word=read_word(x[idx]);
 if(v[idx]&&!sum_bound(word,current,meta.x)){atomic_fetch_min_explicit(error,0xE0000000U,memory_order_relaxed);continue;}
 if(wide_accum(o)){if(v[idx]){wide_accumulate(sum,high,word,o);++count;}}
 else if(wide_pick(o))state_add_wide(sum,high,count,ulong2(word,x[idx+meta.z]),v[idx]!=0,long(row),o);
 else state_add(sum,count,word,DIRECT_FILTERED_REDUCTION&&o.kind==90?true:v[idx]!=0,long(row),current);}
 sums[lane]=sum;counts[lane]=count;highs[lane]=high;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint stride=128;stride;stride/=2){if(lane<stride){
 long a=sums[lane],b=counts[lane],h=highs[lane];if(wide_accum(o)){wide_merge(a,h,sums[lane+stride],highs[lane+stride]);b+=counts[lane+stride];}else if(wide_pick(o))state_merge_wide(a,h,b,sums[lane+stride],highs[lane+stride],counts[lane+stride],o);else state_merge(a,b,sums[lane+stride],counts[lane+stride],o);sums[lane]=a;counts[lane]=b;highs[lane]=h;}
 threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(lane==0)partials[group.y*groups+group.x]=make_partial(sums[0],highs[0],counts[0]);
}
kernel void finish_reduce(device Partial *partials[[buffer(0)]],device Partial *result[[buffer(1)]],
 device const Out *out[[buffer(2)]],constant uint4 &meta[[buffer(3)]],constant ulong &groups[[buffer(4)]],
 uint column[[thread_position_in_grid]]) {
 long sum=0,count=0,high=0;Out o=out[column];
 for(ulong i=0;i<groups;++i){Partial p=partials[column*groups+i];
 if(wide_accum(o)){wide_merge(sum,high,p.total,partial_high(p));count+=p.count;}else if(wide_pick(o))state_merge_wide(sum,high,count,p.total,partial_high(p),p.count,o);else state_merge(sum,count,p.total,p.count,o);}
 if(DIRECT_FILTERED_REDUCTION&&o.kind==90){sum=count;count=1;}
 else state_finish_wide(sum,high,count,o,meta.x);
 result[column]=make_partial(sum,high,count);
}
kernel void order_init(device uint *index[[buffer(0)]],constant uint4 &meta[[buffer(1)]],uint row[[thread_position_in_grid]]){
 if(row<meta.x)index[row]=row;
}
kernel void order_gather(device const T *x[[buffer(0)]],device const uchar *v[[buffer(1)]],
 device T *y[[buffer(2)]],device uchar *w[[buffer(3)]],device const uint *index[[buffer(4)]],
 constant uint4 &meta[[buffer(5)]],constant ulong2 &p[[buffer(6)]],uint row[[thread_position_in_grid]]){
 if(row>=meta.x)return;ulong target=p.x*meta.z+row,source=p.x*meta.z+index[row];y[target]=x[source];w[target]=v[source];
 if(p.y){y[target+meta.z]=x[source+meta.z];w[target+meta.z]=v[source];}
}
)MSL";
  auto shape = validate(r);
  if (shape.radixSort || shape.hashGroups) text += radixSource();
  text += groupSource(r, layout);
  if (shape.hashGroups)
    for (int64_t i = 0; i < r.step_count; ++i) {
      if (!hashGroupCandidate(r, i)) continue;
      const auto &step = r.steps[i];
      auto name = std::to_string(i);
      auto keys = r.code[4 * step.start + 1];
      text += "kernel void group_hash" + name +
        "(device const T *x[[buffer(0)]],device const uchar *v[[buffer(1)]],"
        "device atomic_uint *owners[[buffer(2)]],device ulong *buckets[[buffer(3)]],"
        "constant uint4 &meta[[buffer(4)]],uint row[[thread_position_in_grid]]){"
        "if(row>=meta.x)return;ulong hash=0x9e3779b97f4a7c15UL;";
      for (int64_t j = 0; j < keys; ++j) {
        auto logical = r.code[4 * (step.start + 1 + j) + 1];
        text += "{ulong at=" + std::to_string(layout.physical[logical]) +
          "UL*meta.z+row;hash=key_hash(hash^(v[at]?canonical_key(ulong(x[at])," +
          std::to_string(slotType(r, logical)) + "):0xd6e8feb86659fd93UL));";
        if (slotType(r, logical) == DFM_DECIMAL128)
          text += "if(v[at])hash=key_hash(hash^x[at+meta.z]);";
        text += "}";
      }
      auto capacity = std::to_string(hashCapacity(r.rows)) + "U";
      text += "uint bucket=uint(hash)&(" + capacity + "-1);for(uint probes=0;probes<" + capacity +
        ";){uint owner=atomic_load_explicit(owners+bucket,memory_order_relaxed);"
        "if(owner==0xffffffffU){uint empty=0xffffffffU;"
        "if(atomic_compare_exchange_weak_explicit(owners+bucket,&empty,row,memory_order_relaxed,memory_order_relaxed)){buckets[row]=bucket;return;}"
        "owner=empty;if(owner==0xffffffffU)continue;}if(group_equal" + name +
        "(row,owner,x,v,meta)){buckets[row]=bucket;return;}bucket=(bucket+1)&(" + capacity + "-1);++probes;}}\n";
    }
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
      else if (t == DFM_DECIMAL128)
        text += "device const ulong *raw=(device const ulong*)(bits+p.z);x[at]=raw[2*row];x[at+meta.z]=raw[2*row+1];";
      else
        text += "x[at]=" +
                encode(t, "((device const " + std::string(metalType(t)) +
                              " *)(bits+p.z))[row]") +
                ";";
      text += "v[at]=p.w==ulong(-1)?1:uchar((bits[p.w+bit/8]>>(bit%8))&1);";
      if (t == DFM_DECIMAL128) text += "v[at+meta.z]=v[at];";
      text += "}\n";
    }
  for (int64_t i = 0; i < r.step_count; ++i)
    if (r.steps[i].filter >= 2) {
      auto &step = r.steps[i];
      auto name = std::to_string(i);
      text += "bool order_less" + name +
              "(uint a,uint b,device const T *x,device const uchar *v,constant "
              "uint4 &meta){";
      auto keyCount =
          step.filter == 3 ? r.code[4 * step.start + 1] : step.nodes;
      for (int64_t j = 0; j < keyCount; ++j) {
        auto key = r.code + 4 * (step.start + (step.filter == 4
            ? (j + 1 < keyCount ? j + 1 : 0) : j + (step.filter == 3)));
        auto slot = std::to_string(layout.physical[key[1]]) + "UL*meta.z";
        text += "{ulong ai=" + slot + "+a,bi=" + slot +
                "+b;bool av=v[ai]!=0,bv=v[bi]!=0;if(av!=bv)return " +
                std::string(step.filter == 4 || key[3] ? "av" : "!av") + ";if(av){";
        auto nanFirst = step.filter == 4 && key[0] == DFM_RANK_KEY && key[2] ? "an" : "!an";
        if (slotType(r, key[1]) == DFM_FLOAT32)
          text += "bool "
                  "an=(uint(x[ai])&0x7fffffffU)>0x7f800000U,bn=(uint(x[bi])&"
                  "0x7fffffffU)>0x7f800000U;if(an!=bn)return " + std::string(nanFirst) + ";";
        if (slotType(r, key[1]) == DFM_FLOAT64_STORAGE)
          text += "bool an=f64_nan(ulong(x[ai])),bn=f64_nan(ulong(x[bi]));if(an!=bn)return " + std::string(nanFirst) + ";";
        auto type = std::to_string(slotType(r, key[1])),
             maximum = std::string(key[2] ? "true" : "false");
        if (slotType(r, key[1]) == DFM_DECIMAL128) {
          auto left = std::string(key[2] ? "bi" : "ai"), right = std::string(key[2] ? "ai" : "bi");
          text += "if(wide_less(ulong2(x[" + left + "],x[" + left + "+meta.z]),ulong2(x[" + right + "],x[" + right + "+meta.z])))return true;"
                  "if(wide_less(ulong2(x[" + right + "],x[" + right + "+meta.z]),ulong2(x[" + left + "],x[" + left + "+meta.z])))return false;}}";
        } else
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
  for (int64_t i = 0; i < r.step_count; ++i)
    if (r.steps[i].filter == 4) text += rankSource(r, layout, i);
  if (shape.radixSort)
    for (int64_t i = 0; i < r.step_count; ++i) {
      if (!radixPasses(r, i)) continue;
      auto &step = r.steps[i];
      text += "kernel void sort_radix_keys" + std::to_string(i) +
        "(device const ulong *x[[buffer(0)]],device const uchar *v[[buffer(1)]],"
        "device ulong *keys[[buffer(2)]],constant uint4 &meta[[buffer(3)]],"
        "constant uint2 &p[[buffer(4)]],uint row[[thread_position_in_grid]]){"
        "if(row>=meta.x)return;switch(p.x){";
      for (int64_t j = 0; j < step.nodes; ++j) {
        auto key = r.code + 4 * (step.start + j);
        auto type = slotType(r, key[1]);
        int bits = 8 * width(type);
        auto mask = bits == 64 ? "~0UL" : "((1UL<<" + std::to_string(bits) + ")-1UL)";
        text += "case " + std::to_string(j) + ":{ulong at=" +
          std::to_string(layout.physical[key[1]]) + "UL*meta.z+row;bool valid=v[at]!=0;"
          "if(p.y){keys[row]=valid?" + std::string(key[3] ? "0UL:1UL" : "1UL:0UL") +
          ";}else{ulong word=x[at]&" + mask + ";";
        if (type == DFM_INT8 || type == DFM_INT16 || type == DFM_INT32 || type == DFM_INT64)
          text += "word^=1UL<<" + std::to_string(bits-1) + ";";
        if (key[2]) text += "word^=" + mask + ";";
        text += "keys[row]=valid?word:0UL;}break;}";
      }
      text += "}}\n";
    }
  int64_t first = 0, previousFilter = -1;
  for (auto end : layout.ends) {
    if (r.steps[end].filter >= 2) {
      first = end + 1;
      continue;
    }
    std::array<bool, DFM_MAX_SLOTS> local{};
    text += "kernel void project" + std::to_string(end) +
            "(device T *x[[buffer(0)]],device uchar *v[[buffer(1)]],device "
            "const T *lit[[buffer(2)]],constant uint4 "
            "&meta[[buffer(3)]],device atomic_uint *error[[buffer(4)]],uint "
            "row[[thread_position_in_grid]]){if(row>=meta.x)return;\n";
    if (direct && previousFilter >= 0) {
      auto at = std::to_string(layout.physical[previousFilter]) + "UL*meta.z+row";
      text += "if(!v[" + at + "]||x[" + at + "]==T(0)){";
      if (r.steps[end].filter) {
        auto target = std::to_string(layout.physical[r.steps[end].slot]) +
                      "UL*meta.z+row";
        text += "x[" + target + "]=T(0);v[" + target + "]=0;";
      }
      text += "return;}\n";
    }
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
                    (dtype == DFM_DECIMAL128 ? "ulong2(x[" + idx + "],x[(" + idx + ")+meta.z])" : typed(r) ? decode(dtype, "x[" + idx + "]")
                              : "x[" + idx + "]") +
                    ";" + valid + "=v[" + idx + "]!=0;";
          }
        } else if (op == DFM_SLICE) {
          text += "{long off=long(lit[" + std::to_string(c[1]) +
                  "]),length=long(lit[" + std::to_string(c[2]) +
                  "]),n=long(meta.x);long begin=off<0?max(n+off,0L):min(off,n);"
                  "long count=length==-1?n-begin:min(length,n-begin);" + z +
                  "=uchar(long(row)>=begin&&long(row)-begin<count);}";
        } else if (op == DFM_CAST) {
          text += valid + "=" + av + ";if(" + valid + "){" +
                  castExpression(r.node_types[s.start + c[1]], dtype, a, z,
                                 valid, c[3] != 0, s.start + j + 1) +
                  "}";
        } else if (op == DFM_INT_IS_IN) {
          auto operand = nodeType(r, s.start + c[1]);
          text += valid + "=" + av + ";if(" + valid + "){bool hit=false;";
          text += "for(ulong member=0;member<" + std::to_string(c[3]) +
                  "UL;++member)hit|=(" + a + "==" +
                  decode(operand, "lit[" + std::to_string(c[2]) + "UL+member]") +
                  ");" + z + "=uchar(hit);}";
        } else if (op == DFM_LITERAL_NULL)
          text += valid + "=false;";
        else if (op == DFM_LITERAL_INT && dtype == DFM_DECIMAL128)
          text += z + "=ulong2(lit[" + std::to_string(s.start+j) + "],lit[" + std::to_string(c[3]) + "]);";
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
          auto nan = dtype == DFM_FLOAT64_STORAGE ? "f64_nan(" + a + ")" :
              "((as_type<uint>(" + a + ")&0x7fffffffU)>0x7f800000U)";
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
          auto operandType = nodeType(r, s.start + c[1]);
          if (op == DFM_DIV) {
            text += z + "=f64_div(f64_word(" + encode(operandType,a) + "," + std::to_string(operandType) +
                    "),f64_word(" + encode(operandType,b) + "," + std::to_string(operandType) + "));";
          } else if (op == DFM_SQRT) {
            auto expression = "f64_sqrt(f64_word(" + encode(operandType,a) + "," + std::to_string(operandType) + "))";
            text += z + "=" + (dtype == DFM_FLOAT32
                ? "as_type<float>(f64_to_float32(" + expression + "))"
                : expression) + ";";
          } else if (op == DFM_FLOORDIV && operandType == DFM_FLOAT32) {
            text += z + "=as_type<float>(f32_floor_div(as_type<uint>(" + a + "),as_type<uint>(" + b + ")));";
          } else if (op == DFM_MOD && operandType == DFM_FLOAT32) {
            text += z + "=as_type<float>(f32_mod(as_type<uint>(" + a + "),as_type<uint>(" + b + ")));";
          } else if (operandType == DFM_FLOAT64_STORAGE) {
            std::string expression;
            if (op == DFM_ADD || op == DFM_SUB || op == DFM_MUL)
              expression = std::string(op == DFM_ADD ? "f64_add(" : op == DFM_SUB ? "f64_sub(" : "f64_mul(") + a + "," + b + ")";
            else if (op == DFM_NEG) expression = "(" + a + "^0x8000000000000000UL)";
            else if (op == DFM_ABS) expression = "f64_abs(" + a + ")";
            else if (op == DFM_FLOOR || op == DFM_CEIL) expression = "f64_integral(" + a + "," + std::to_string(op) + ")";
            else if (op == DFM_ROUND)
              expression = c[3] == 0 ? "f64_round(" + a + ")"
                  : "f64_decimal_round(" + a + ",lit[" + std::to_string(s.start+j) + "]," + (c[3] < 0 ? "true" : "false") + ")";
            else if (op == DFM_FLOORDIV) expression = "f64_integral(f64_div(" + a + "," + b + "),55)";
            else if (op == DFM_MOD) expression = "f64_mod(" + a + "," + b + ")";
            else if (op == DFM_IS_NAN) expression = "f64_nan(" + a + ")";
            else if (op == DFM_IS_NOT_NAN) expression = "!f64_nan(" + a + ")";
            else if (op == DFM_IS_FINITE) expression = "(f64_abs(" + a + ")<0x7ff0000000000000UL)";
            else if (op == DFM_IS_INFINITE) expression = "f64_inf(" + a + ")";
            else if (op == DFM_EQ || op == DFM_NE)
              expression = std::string(op == DFM_NE ? "!" : "") + "f64_equal(" + a + "," + b + ")";
            else if (op == DFM_GT || op == DFM_LT || op == DFM_GE || op == DFM_LE) {
              expression = (op == DFM_GT || op == DFM_GE ? "f64_less(" + b + "," + a : "f64_less(" + a + "," + b) + ")";
              if (op == DFM_GE || op == DFM_LE) expression += "||f64_equal(" + a + "," + b + ")";
            } else if (op == DFM_CLIP_LOW || op == DFM_CLIP_HIGH)
              expression = (op == DFM_CLIP_LOW ? "f64_less(" + a + "," + b : "f64_less(" + b + "," + a) + ")?" + b + ":" + a;
            else fail("Unsupported emulated Float64 expression");
            text += z + "=" + typename_ + "(" + expression + ");";
          } else if (op >= DFM_DT_YEAR && op <= DFM_DT_TRUNCATE) {
            const int64_t perSecond = c[3] == 1 ? 1000 : c[3] == 2 ? 1000000 : 1000000000;
            const int64_t dayTicks = c[3] == 0 ? 1 : perSecond * 86400;
            const auto perDay = std::to_string(dayTicks) + "L";
            const auto second = std::to_string(perSecond) + "L";
            if (c[3] == 0)
              text += "long days=" + a + ";";
            else
              text += "long days=" + a + "/" + perDay + ";if(" + a + "%" + perDay + "<0)--days;";
            if (op == DFM_DT_TRUNCATE) {
              const int64_t origin = c[2] % (7 * dayTicks) == 0 ? 4 * dayTicks : 0;
              const auto step = std::to_string(c[2]);
              text += "long shifted=as_type<long>(as_type<ulong>(" + a + ")-" + std::to_string(origin) + "UL);long quotient=shifted/" + step + "L;if(shifted%" + step + "L<0)--quotient;" + z + "=as_type<long>(as_type<ulong>(quotient)*" + step + "UL+" + std::to_string(origin) + "UL);";
            } else if (op == DFM_DT_DATE)
              text += z + "=days;";
            else if (op == DFM_DT_WEEKDAY) {
              // Explicit unsigned wrapping matches the CPU calendar's Int64
              // addition even at the extremes of stored date values.
              text += "long wd=as_type<long>(as_type<ulong>(days)+3UL)%7L;if(wd<0)wd+=7L;" + z + "=wd+1L;";
            } else if ((op >= DFM_DT_HOUR && op <= DFM_DT_NANOSECOND) || op == DFM_DT_TIME) {
              text += "long within=" + a + "%" + perDay + ";if(within<0)within+=" + perDay + ";long seconds=within/" + second + ";";
              if (op == DFM_DT_HOUR) text += z + "=seconds/3600L;";
              else if (op == DFM_DT_MINUTE) text += z + "=(seconds/60L)%60L;";
              else if (op == DFM_DT_SECOND) text += z + "=seconds%60L;";
              else if (op == DFM_DT_TIME) text += z + "=within*" + std::to_string(1000000000 / perSecond) + "L;";
              else text += z + "=(within%" + second + ")*" + std::to_string(1000000000 / perSecond) + "L;";
            } else if (op == DFM_DT_ORDINAL_DAY && c[3] == 4) {
              text += z + "=days+1L;";
            } else {
              text += "long shifted=as_type<long>(as_type<ulong>(days)+719468UL);long era=shifted/146097L;if(shifted%146097L<0)--era;"
                      "long doe=as_type<long>(as_type<ulong>(shifted)-as_type<ulong>(era)*146097UL);"
                      "long yoe=(doe-doe/1460L+doe/36524L-doe/146096L)/365L;"
                      "long doy=doe-(365L*yoe+yoe/4L-yoe/100L);long mp=(5L*doy+2L)/153L;"
                      "long day=doy-(153L*mp+2L)/5L+1L;long month=mp<10L?mp+3L:mp-9L;long year=yoe+era*400L+(month<=2L?1L:0L);";
              if (op == DFM_DT_YEAR) text += z + "=year;";
              else if (op == DFM_DT_MONTH) text += z + "=month;";
              else if (op == DFM_DT_DAY) text += z + "=day;";
              else text += "bool leap=(year%4L==0L&&year%100L!=0L)||year%400L==0L;" + z + "=(367L*month-362L)/12L+day-(month<=2L?0L:leap?1L:2L);";
            }
          } else if (op >= DFM_IS_NAN && op <= DFM_IS_INFINITE) {
            auto bits = "(as_type<uint>(" + a + ")&0x7fffffffU)";
            auto test = op == DFM_IS_NAN       ? bits + ">0x7f800000U"
                        : op == DFM_IS_NOT_NAN ? bits + "<=0x7f800000U"
                        : op == DFM_IS_FINITE  ? bits + "<0x7f800000U"
                                               : bits + "==0x7f800000U";
            text += z + "=uchar(" + test + ");";
          } else if (op == DFM_ROUND && operandType == DFM_FLOAT32) {
            auto wide = "f64_from_float32(as_type<uint>(" + a + "))";
            auto rounded = c[3] == 0 ? "f64_round(" + wide + ")"
                : "f64_decimal_round(" + wide + ",lit[" + std::to_string(s.start+j) + "]," + (c[3] < 0 ? "true" : "false") + ")";
            text += z + "=as_type<float>(f64_to_float32(" + rounded + "));";
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
            if (op != DFM_NOT && op != DFM_XOR &&
                (nodeType(r,s.start+c[1]) == DFM_DECIMAL128 || nodeType(r,s.start+c[2]) == DFM_DECIMAL128)) {
              auto left = nodeType(r,s.start+c[1]) == DFM_DECIMAL128 ? a : "wide_word(long(" + a + "))";
              auto right = nodeType(r,s.start+c[2]) == DFM_DECIMAL128 ? b : "wide_word(long(" + b + "))";
              expr = "wide_compare(" + left + "," + right + "," + std::to_string(op) + ")";
            }
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
                (r.slot_types && r.slot_types[s.slot] == DFM_DECIMAL128 ? "slot" + slot + ".x" : typed(r) ? encode(r.slot_types[s.slot], "slot" + slot)
                          : "slot" + slot) +
                ";v[" + idx + "]=uchar(valid" + slot + ");\n";
        if (typed(r) && r.slot_types[s.slot] == DFM_DECIMAL128)
          text += "x[(" + idx + ")+meta.z]=slot" + slot + ".y;v[(" + idx + ")+meta.z]=uchar(valid" + slot + ");\n";
      }
    }
    text += "}\n";
    if (direct && r.steps[end].filter)
      previousFilter = r.steps[end].slot;
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
      fail("Metal pipeline creation (" + std::string(name.UTF8String) + "): " +
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
  explicit Buffers(Context &context) : c(context) {
    dfm_context_retain(&c);
  }
  ~Buffers() {
    c.recycle(all);
    dfm_context_release(&c);
  }
  Buffers(const Buffers &) = delete;
  Buffers &operator=(const Buffers &) = delete;
  void transfer(id<MTLBuffer> value, Buffers &to) {
    auto it = std::find(all.begin(), all.end(), value);
    if (it == all.end())
      fail("Metal resident buffer owner mismatch");
    to.all.push_back(value);
    to.requested = add(to.requested, int64_t(value.length));
    all.erase(it);
  }
  id<MTLBuffer> make(int64_t bytes) {
    auto b = c.acquire(std::max<int64_t>(1, bytes));
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
struct WidePartial { int64_t total, high, count, padding; };
WidePartial loadPartial(id<MTLBuffer> buffer, int64_t index, int64_t stride) {
  if (stride == 32) return static_cast<WidePartial *>(buffer.contents)[index];
  auto p = static_cast<Partial *>(buffer.contents)[index];
  return {p.total, 0, p.count, 0};
}
struct Query {
  std::atomic<int64_t> refs{1}, live{0}, peak{0};
  Context &context;
  int64_t budget;
  Query(Context &c, int64_t limit) : context(c), budget(limit) {
    dfm_context_retain(&c);
  }
  ~Query() { dfm_context_release(&context); }
  void check(int64_t additional) {
    auto total = add(live.load(), additional);
    if (budget >= 0 && total > budget)
      fail("Metal resident query exceeds memory budget before submission");
    auto previous = peak.load();
    while (previous < total && !peak.compare_exchange_weak(previous, total)) {}
  }
};
void queryRelease(Query *q) {
  if (q && --q->refs == 0)
    delete q;
}
struct ResidentColumn {
  int64_t slot, type, reduction, minimum;
};
struct Resident {
  Query &query;
  Buffers buffers;
  id<MTLBuffer> values = nil, validity = nil, bitmap = nil;
  int64_t rows = 0, cap = 0, packed = 0, retained = 0;
  bool scalar = false, final = false;
  int64_t scalar_stride = 16;
  std::vector<ResidentColumn> columns;
  DFMStats stats{};
  explicit Resident(Query &q) : query(q), buffers(q.context) { ++q.refs; }
  ~Resident() {
    query.live -= retained;
    queryRelease(&query);
  }
  void seal() {
    retained = buffers.requested;
    query.live += retained;
  }
};
struct ResidentCopy {
  uint64_t source, validity, target, rows;
  int64_t scalar, minimum, sum, padding;
  uint64_t source_high, target_high;
};
Resident *residentInput(Query &query, const DFMResidentInput *refs,
                        int64_t index, int64_t rows, int64_t type) {
  if (!refs || !refs[index].table)
    return nullptr;
  auto &t = *static_cast<Resident *>(refs[index].table);
  auto col = refs[index].column;
  if (&t.query != &query || t.rows != rows || col < 0 ||
      col >= int64_t(t.columns.size()) || t.columns[col].type != type)
    fail("Metal resident input context, rows, column or dtype mismatch");
  return &t;
}
ResidentCopy residentCopy(const Resident &table, int64_t column,
                          int64_t target, int64_t target_cap = 0) {
  auto c = table.columns[column];
  bool wide = c.type == DFM_DECIMAL128;
  auto source = table.scalar ? (table.scalar_stride / 8) * c.slot : c.slot * table.cap;
  return {uint64_t(source),
          uint64_t(table.scalar ? table.scalar_stride * c.slot + (table.scalar_stride == 32 ? 16 : 8) : c.slot * table.cap),
          uint64_t(target), uint64_t(table.rows), table.scalar, c.minimum,
          c.reduction == DFM_SUM || c.reduction == DFM_WIDE_SUM || c.reduction == DFM_WIDE_INTEGER_SUM, 0,
          uint64_t(wide ? source + (table.scalar ? 1 : table.cap) : 0),
          uint64_t(wide ? target + target_cap : 0)};
}
void run(Context &c, const DFMRequest &r, DFMStats &stats,
         Query *query = nullptr, const DFMResidentInput *refs = nullptr,
         Resident **resident = nullptr, bool final = false) {
  auto shape = validate(r, true, refs, query != nullptr);
  if (query && !typed(r))
    fail("Metal resident row requests require typed word storage");
  if (query)
    for (int64_t i = 0; i < r.input_count; ++i)
      residentInput(*query, refs, i, r.rows, r.inputs[i].dtype);
  if (query && !final)
    shape.outputBits = 1;
  auto layout = Layout(r);
  stats = {};
  stats.gpu_ns = -1;
  stats.memory = estimate(r, shape, refs, query != nullptr, final);
  if (query)
    query->check(stats.memory.peak_bytes);
  int64_t budget = c.headroom();
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
  RadixShape radixShape(shape.cap);
  if (shape.radixSort || shape.hashGroups)
    for (auto bytes : {mul(shape.cap, 8), mul(shape.cap, 4), radixShape.histogram, int64_t(64), int64_t(32)})
      if (uint64_t(bytes) > c.device.maxBufferLength)
        fail("Metal radix buffer exceeds maximum buffer length before submission");
  if (shape.hashGroups && uint64_t(mul(hashCapacity(shape.cap), 4)) > c.device.maxBufferLength)
    fail("Metal hash table exceeds maximum buffer length before submission");
  auto src = source(r);
  bool hit = false;
  auto start = Clock::now();
  auto p = pipelines(c, src, hit);
  stats.pipeline_cache_hit = hit;
  stats.pipeline_compile_ns = hit ? 0 : ns(start);
  Buffers b{c};
  auto x = b.make(shape.matrix), v = b.make(shape.validity);
  id<MTLBuffer> y = nil, w = nil, rank = nil, blocks = nil;
  if (shape.filter || shape.order) {
    y = b.make(shape.matrix);
    w = b.make(shape.validity);
    rank = b.make(mul(shape.cap, 4));
    blocks = b.make(mul(shape.order ? shape.cap : shape.blocks, 4));
  }
  id<MTLBuffer> radixKeys = nil, radixRanks = nil, radixHistogram = nil, radixTotals = nil, radixArgs = nil;
  if (shape.radixSort || shape.hashGroups) {
    radixKeys = b.make(mul(shape.cap, 8)); radixRanks = b.make(mul(shape.cap, 4));
    radixHistogram = b.make(radixShape.histogram); radixTotals = b.make(64); radixArgs = b.make(32);
  }
  id<MTLBuffer> hashOwners = shape.hashGroups ? b.make(mul(hashCapacity(shape.cap), 4)) : nil;
  id<MTLBuffer> groupArgs = shape.grouped ? b.make(16) : nil;
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
    if (refs && refs[i].table) {
      unpack.push_back({});
      continue;
    }
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
    if (op == DFM_ROUND && r.code[4*i+3] != 0 &&
        (dtype == DFM_FLOAT32 || dtype == DFM_FLOAT64_STORAGE)) {
      memcpy(static_cast<char *>(literal.contents) + i * 8, r.literals + i, 8);
      continue;
    }
    if (op == DFM_GROUP_OUTPUT) {
      memcpy(static_cast<char *>(literal.contents) + i * 8, &integer, 8);
      continue;
    }
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
                      slotType(r, r.outputs[i].slot)};
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
  auto indirect = [&](const std::string &name) {
    [encoder setComputePipelineState:p->kernels.at(name)];
    [encoder dispatchThreadgroupsWithIndirectBuffer:radixArgs indirectBufferOffset:16
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++launches;
  };
  auto radixDigit = [&](id<MTLBuffer> __strong &orderA, id<MTLBuffer> __strong &orderB, uint32_t shift) {
    buffer(0, orderA); buffer(1, radixKeys); buffer(2, radixRanks);
    buffer(3, radixHistogram); buffer(4, radixArgs); bytes(5, &shift, 4);
    indirect("radix_rank");
    buffer(0, radixHistogram); buffer(1, radixTotals); buffer(2, radixArgs);
    dispatch("radix_prefix", 16 * 256, 1, true);
    buffer(0, orderA); buffer(1, orderB); buffer(2, radixKeys);
    buffer(3, radixRanks); buffer(4, radixHistogram); buffer(5, radixTotals);
    buffer(6, radixArgs); bytes(7, &shift, 4);
    indirect("radix_scatter"); std::swap(orderA, orderB);
  };
  offset = 0;
  for (int64_t i = 0; i < r.input_count; ++i) {
    if (refs && refs[i].table) {
      auto &t = *static_cast<Resident *>(refs[i].table);
      auto params = residentCopy(t, refs[i].column,
                                layout.physical[i] * shape.cap, shape.cap);
      buffer(0, t.values);
      buffer(1, t.validity);
      buffer(2, x);
      buffer(3, v);
      bytes(4, &params, sizeof(params));
      dispatch("resident_copy", shape.cap);
      continue;
    }
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
    if (step.filter >= 2) {
      auto orderA = rank, orderB = blocks;
      buffer(0, orderA);
      buffer(1, meta);
      dispatch("order_init", shape.cap);
      bool hashGroup = shape.hashGroups && hashGroupCandidate(r, i);
      if (hashGroup) {
        uint32_t capacity = hashCapacity(shape.cap), capacityBlocks = radixShape.blocks;
        buffer(0, hashOwners); bytes(1, &capacity, 4);
        dispatch("hash_owners_init", capacity);
        buffer(0, meta); buffer(1, radixArgs); bytes(2, &capacityBlocks, 4);
        dispatch("radix_args", 1);
        buffer(0, x); buffer(1, v); buffer(2, hashOwners); buffer(3, radixKeys); buffer(4, meta);
        indirect("group_hash" + std::to_string(i));
        for (uint32_t digit = 0; digit < hashGroupPasses(r); ++digit)
          radixDigit(orderA, orderB, digit * 4);
      } else
      if (shape.radixSort && radixPasses(r, i)) {
        uint32_t capacityBlocks = radixShape.blocks;
        buffer(0, meta); buffer(1, radixArgs); bytes(2, &capacityBlocks, 4);
        dispatch("radix_args", 1);
        for (int64_t key = step.nodes; key-- > 0;) {
          for (uint32_t phase = 0; phase < 2; ++phase) {
            uint32_t params[2] = {uint32_t(key), phase};
            buffer(0, x); buffer(1, v); buffer(2, radixKeys); buffer(3, meta); bytes(4, params, 8);
            indirect("sort_radix_keys" + std::to_string(i));
            auto type = slotType(r, r.code[4 * (step.start + key) + 1]);
            uint32_t bits = phase ? 4 : 8 * width(type);
            for (uint32_t shift = 0; shift < bits; shift += 4)
              radixDigit(orderA, orderB, shift);
          }
        }
      } else
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
      if (step.filter == 4) {
        auto name = std::to_string(i);
        auto h = r.code + 4 * step.start;
        if (h[3] == 3) {
          buffer(0, x); buffer(1, v); buffer(2, orderA); buffer(3, y);
          buffer(4, orderB); buffer(5, meta);
          dispatch("rank_prepare" + name, shape.cap, 1, true);
          buffer(0, orderB); buffer(1, meta);
          dispatch("rank_offsets" + name, 1);
        }
        buffer(0, x); buffer(1, v); buffer(2, orderA); buffer(3, y);
        buffer(4, orderB); buffer(5, meta);
        dispatch("rank_assign" + name, shape.cap);
        continue;
      }
      if (step.filter == 3) {
        auto name = std::to_string(i);
        buffer(0, x);
        buffer(1, v);
        buffer(2, orderA);
        buffer(3, meta);
        dispatch("group_mark" + name, shape.cap);
        auto h = r.code + 4 * step.start;
        uint64_t flag = layout.physical[h[2]];
        buffer(0, x);
        buffer(1, v);
        buffer(2, rank);
        buffer(3, blocks);
        buffer(4, meta);
        bytes(5, &flag, 8);
        dispatch("rank_rows", shape.cap, 1, true);
        buffer(0, blocks);
        buffer(1, meta);
        dispatch("group_offsets" + name, 1);
        buffer(0, x);
        buffer(1, v);
        buffer(2, rank);
        buffer(3, blocks);
        buffer(4, meta);
        buffer(5, groupArgs);
        dispatch("group_starts" + name, shape.cap);
        buffer(0, x);
        buffer(1, v);
        buffer(2, y);
        buffer(3, w);
        buffer(4, meta);
        buffer(5, arithmeticError);
        buffer(6, literal);
        [encoder setComputePipelineState:p->kernels.at("group_reduce" + name)];
        [encoder dispatchThreadgroupsWithIndirectBuffer:groupArgs
                                   indirectBufferOffset:0
                                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ++launches;
        std::swap(x, y);
        std::swap(v, w);
        if (hashGroup) {
          uint32_t capacityBlocks = radixShape.blocks;
          buffer(0, meta); buffer(1, radixArgs); bytes(2, &capacityBlocks, 4);
          dispatch("radix_args", 1);
          orderA = rank; orderB = blocks;
          buffer(0, orderA); buffer(1, meta); indirect("order_init");
          for (int64_t stride = 1; stride < shape.cap; stride *= 2) {
            uint32_t span = stride;
            buffer(0, x); buffer(1, v); buffer(2, orderA); buffer(3, orderB);
            buffer(4, meta); bytes(5, &span, 4);
            indirect("order_merge" + name); std::swap(orderA, orderB);
          }
          for (int64_t logical = 0; logical < r.slots; ++logical) {
            if (!layout.group_after[i][logical]) continue;
            uint64_t slot[2] = {uint64_t(layout.physical[logical]), uint64_t(slotType(r, logical) == DFM_DECIMAL128)};
            buffer(0, x); buffer(1, v); buffer(2, y); buffer(3, w);
            buffer(4, orderA); buffer(5, meta); bytes(6, slot, 16);
            indirect("order_gather");
          }
          std::swap(x, y); std::swap(v, w);
        }
        continue;
      }
      for (int64_t j = 0; j < step.gather_count; ++j) {
        auto logical = r.gathers[step.gather_start + j];
        auto physical = layout.physical[logical];
        if (!layout.gathered[i][logical])
          continue;
        uint64_t slot[2] = {uint64_t(physical), uint64_t(slotType(r, logical) == DFM_DECIMAL128)};
        buffer(0, x);
        buffer(1, v);
        buffer(2, y);
        buffer(3, w);
        buffer(4, orderA);
        buffer(5, meta);
        bytes(6, slot, 16);
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
    if (step.filter && !directFilteredReduction(r)) {
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
        auto logical = r.gathers[step.gather_start + j];
        auto physical = layout.physical[logical];
        if (!layout.gathered[i][logical])
          continue;
        uint64_t params[4] = {slot, static_cast<uint64_t>(physical), uint64_t(slotType(r, logical) == DFM_DECIMAL128), 0};
        buffer(0, x);
        buffer(1, v);
        buffer(2, y);
        buffer(3, w);
        buffer(4, rank);
        buffer(5, blocks);
        buffer(6, meta);
        bytes(7, params, 32);
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
    buffer(6, arithmeticError);
    dispatch("reduce_rows", groups * 256, r.output_count, true);
  }
  id<MTLBuffer> result = nil;
  if (r.reductions) {
    result = b.make(mul(r.output_count, partialWidth(r)));
    buffer(0, partial);
    buffer(1, result);
    buffer(2, out);
    buffer(3, meta);
    bytes(4, &groups, 8);
    dispatch("finish_reduce", r.output_count);
  } else if (!query || final) {
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
    if (fault == 0xE0000000U)
      fail("Metal unsupported [accumulator]: cannot prove exact wide accumulation "
           "within the native Int64/UInt64 range");
    if (fault == 0xF0000000U)
      fail("Integer overflow in Metal grouped sum");
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
    for (int64_t i = 0; i < r.output_count; ++i) {
      auto value = loadPartial(result, i, partialWidth(r));
      if (r.outputs[i].reduction == DFM_WIDE_INTEGER_SUM &&
          value.count >= r.outputs[i].min_count &&
          (r.outputs[i].dtype == DFM_UINT64 ? value.high != 0
              : value.high != (value.total < 0 ? -1 : 0)))
        fail("Integer overflow in Metal sum");
      if (r.outputs[i].reduction == DFM_SUM &&
          value.count >= r.outputs[i].min_count &&
          ((r.outputs[i].dtype == DFM_INT32 &&
            (value.total < INT32_MIN || value.total > INT32_MAX)) ||
           (r.outputs[i].dtype == DFM_UINT32 &&
            (value.total < 0 || uint64_t(value.total) > UINT32_MAX))))
        fail("Integer overflow in Metal sum");
    }
  }
  if (query) {
    auto t = std::make_unique<Resident>(*query);
    t->rows = rows;
    t->cap = shape.cap;
    t->packed = shape.packed;
    t->scalar = r.reductions;
    t->scalar_stride = partialWidth(r);
    t->final = final;
    for (int64_t i = 0; i < r.output_count; ++i) {
      auto o = r.outputs[i];
      t->columns.push_back({r.reductions ? i : layout.physical[o.slot],
                             o.dtype, o.reduction, o.min_count});
    }
    if (r.reductions) {
      t->values = result;
      t->validity = result;
      b.transfer(result, t->buffers);
    } else {
      t->values = x;
      t->validity = v;
      b.transfer(x, t->buffers);
      b.transfer(v, t->buffers);
      if (final) {
        t->bitmap = bits;
        b.transfer(bits, t->buffers);
      }
    }
    // Cardinality and the error word are synchronization metadata. Intermediate
    // column payloads stay in Metal storage; checked sums also inspect scalars.
    stats.download_bytes = 8;
    if (r.reductions && rows)
      for (int64_t i = 0; i < r.output_count; ++i)
        if (r.outputs[i].reduction == DFM_SUM ||
            r.outputs[i].reduction == DFM_WIDE_INTEGER_SUM)
          stats.download_bytes += partialWidth(r);
    stats.result_copy_ns = 0;
    t->stats = stats;
    t->seal();
    *resident = t.release();
    return;
  }
  for (int64_t i = 0; i < r.output_count; ++i) {
    auto &o = r.outputs[i];
    if (r.reductions) {
      if (!rows)
        continue;
      auto a = loadPartial(result, i, partialWidth(r));
      bool valid =
          (o.reduction == DFM_SUM || o.reduction == DFM_WIDE_SUM || o.reduction == DFM_WIDE_INTEGER_SUM) ? a.count >= o.min_count : a.count > 0;
      if (!valid)
        a.total = a.high = 0;
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
                 words + row, std::min<int64_t>(8, width(o.dtype)));
        if (o.dtype == DFM_DECIMAL128)
          for (int64_t row = 0; row < rows; ++row)
            memcpy(static_cast<char *>(o.values) + row * 16 + 8, words + shape.cap + row, 8);
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
int32_t dfm_probe(int64_t device_id, const DFMRequest *request,
                  DFMProbeInfo *info, char **err) {
  if (err) *err = nullptr;
  @autoreleasepool {
    try {
      if (!info) fail("Missing Metal metadata probe output");
      if (request) validate(*request);
      auto devices = MTLCopyAllDevices();
      if (device_id < 0 || device_id >= int64_t(devices.count))
        fail("Metal device index unavailable");
      id<MTLDevice> device = devices[device_id];
      int64_t cached = 0, idle = 0;
      {
        std::lock_guard<std::mutex> lock(defaultMutex);
        if (device_id == 0 && defaultContext) {
          auto &c = *defaultContext;
          {
            std::lock_guard<std::mutex> workspaceLock(c.workspace_mutex);
            idle = c.idle_bytes;
          }
          if (request) {
            auto src = source(*request);
            std::lock_guard<std::mutex> cacheLock(c.mutex);
            cached = c.cache.count(src) ? 1 : 0;
          }
        }
      }
      char model[256] = {}, hostname[256] = {};
      size_t modelSize = sizeof(model);
      if (sysctlbyname("hw.model", model, &modelSize, nullptr, 0) != 0 ||
          gethostname(hostname, sizeof(hostname) - 1) != 0)
        fail("Cannot identify Apple calibration host");
      auto process = NSProcessInfo.processInfo;
      static thread_local std::string host;
      host = std::string(hostname) + ";model=" + model +
          ";memory=" + std::to_string(process.physicalMemory) +
          ";processors=" + std::to_string(process.activeProcessorCount) +
          ";os=" + process.operatingSystemVersionString.UTF8String +
          ";lowPower=" + std::to_string(process.lowPowerModeEnabled) +
          ";thermal=" + std::to_string(process.thermalState);
      *info = {device.registryID, int64_t(device.recommendedMaxWorkingSetSize),
               int64_t(device.currentAllocatedSize), int64_t(device.hasUnifiedMemory),
               cached, idle, int64_t(device.maxBufferLength), host.c_str()};
      return 0;
    } catch (const std::exception &e) {
      error(err, e);
      return 1;
    }
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
      ctx->workspace_limit = std::min<int64_t>(256 * 1024 * 1024,
          int64_t(ctx->device.recommendedMaxWorkingSetSize / 16));
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
int32_t dfm_workspace_info(void *p, DFMWorkspaceInfo *info, char **err) {
  if (err)
    *err = nullptr;
  try {
    if (!p || !info)
      fail("Missing Metal workspace context or info");
    auto &c = *static_cast<Context *>(p);
    std::lock_guard<std::mutex> lock(c.workspace_mutex);
    *info = {c.idle_bytes, int64_t(c.idle.size()), c.workspace_hits,
             c.workspace_allocations, c.workspace_limit};
    return 0;
  } catch (const std::exception &e) {
    error(err, e);
    return 1;
  }
}
int32_t dfm_workspace_trim(void *p, char **err) {
  if (err)
    *err = nullptr;
  try {
    if (!p)
      fail("Missing Metal workspace context");
    auto &c = *static_cast<Context *>(p);
    std::lock_guard<std::mutex> lock(c.workspace_mutex);
    c.idle.clear();
    c.idle_bytes = 0;
    return 0;
  } catch (const std::exception &e) {
    error(err, e);
    return 1;
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
int64_t dfm_plan_features(const DFMRequest *r, char **err) {
  if (err)
    *err = nullptr;
  try {
    if (!r)
      fail("Missing Metal plan feature request");
    auto shape = validate(*r);
    return (directFilteredReduction(*r) ? 1 : 0) | (shape.radixSort ? 4 : 0) | (shape.hashGroups ? 2 : 0);
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
int64_t dfm_resident_abi_version() { return DFM_RESIDENT_ABI_VERSION; }
void *dfm_query_create(void *p, int64_t budget, char **err) {
  if (err)
    *err = nullptr;
  try {
    if (!p || budget < -1)
      fail("Invalid Metal resident query context or budget");
    return new Query(*static_cast<Context *>(p), budget);
  } catch (const std::exception &e) {
    error(err, e);
    return nullptr;
  }
}
void dfm_query_release(void *p) { queryRelease(static_cast<Query *>(p)); }
int32_t dfm_query_info(void *p, DFMQueryInfo *info, char **err) {
  if (err)
    *err = nullptr;
  try {
    if (!p || !info)
      fail("Missing Metal resident query or info");
    auto &q = *static_cast<Query *>(p);
    *info = {q.live.load(), q.peak.load(), q.budget};
    return 0;
  } catch (const std::exception &e) {
    error(err, e);
    return 1;
  }
}
int32_t dfm_resident_rows(void *p, const DFMRequest *r,
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
        fail("Invalid Metal resident row request");
      auto &q = *static_cast<Query *>(p);
      std::lock_guard<std::mutex> lock(q.context.mutex);
      Resident *result = nullptr;
      run(q.context, *r, *stats, &q, refs, &result, final);
      *table = result;
      return 0;
    } catch (const std::exception &e) {
      error(err, e);
      return 1;
    }
  }
}
int32_t dfm_resident_info(void *p, DFMResidentInfo *info, char **err) {
  if (err)
    *err = nullptr;
  try {
    if (!p || !info)
      fail("Missing Metal resident table or info");
    auto &t = *static_cast<Resident *>(p);
    *info = {t.rows, int64_t(t.columns.size()), t.retained};
    return 0;
  } catch (const std::exception &e) {
    error(err, e);
    return 1;
  }
}
int32_t dfm_resident_export(void *p, const DFMOutput *outputs, int64_t count,
                           DFMStats *stats, char **err) {
  if (err)
    *err = nullptr;
  @autoreleasepool {
    try {
      if (!p || !outputs || !stats)
        fail("Missing Metal resident export table, outputs or stats");
      auto &t = *static_cast<Resident *>(p);
      std::lock_guard<std::mutex> lock(t.query.context.mutex);
      if (!t.final || count != int64_t(t.columns.size()))
        fail("Metal resident table was not prepared for final export");
      auto packed = (t.rows + 7) / 8;
      int64_t payload = 0;
      for (int64_t i = 0; i < count; ++i) {
        auto o = outputs[i];
        if (o.dtype != t.columns[i].type ||
            (t.rows && (!o.values || !o.validity)))
          fail("Metal resident export output metadata mismatch");
        payload = add(payload, add(packed, o.dtype == DFM_BOOL ? packed :
                                                        mul(t.rows, width(o.dtype))));
      }
      t.query.check(payload);
      auto start = Clock::now();
      for (int64_t i = 0; i < count; ++i) {
        auto o = outputs[i];
        auto c = t.columns[i];
        if (!t.rows)
          continue;
        if (t.scalar) {
          auto a = loadPartial(t.values, c.slot, t.scalar_stride);
          bool valid = (c.reduction == DFM_SUM || c.reduction == DFM_WIDE_SUM || c.reduction == DFM_WIDE_INTEGER_SUM) ? a.count >= c.minimum : a.count > 0;
          if (!valid)
            a.total = a.high = 0;
          memcpy(o.values, &a.total, width(o.dtype));
          o.validity[0] = valid;
        } else {
          auto *bitmap = static_cast<uint8_t *>(t.bitmap.contents) +
                         2 * i * t.packed;
          memcpy(o.validity, bitmap, packed);
          if (o.dtype == DFM_BOOL) {
            memcpy(o.values, bitmap + t.packed, packed);
          } else {
            auto *words = static_cast<uint64_t *>(t.values.contents) +
                          c.slot * t.cap;
            auto bytes = width(o.dtype);
            for (int64_t row = 0; row < t.rows; ++row)
              memcpy(static_cast<char *>(o.values) + row * bytes, words + row, std::min<int64_t>(8, bytes));
            if (o.dtype == DFM_DECIMAL128)
              for (int64_t row = 0; row < t.rows; ++row)
                memcpy(static_cast<char *>(o.values) + row * 16 + 8, words + t.cap + row, 8);
          }
        }
      }
      *stats = t.stats;
      stats->download_bytes = add(stats->download_bytes, payload);
      stats->result_copy_ns = ns(start);
      return 0;
    } catch (const std::exception &e) {
      error(err, e);
      return 1;
    }
  }
}
void dfm_resident_release(void *p) { delete static_cast<Resident *>(p); }
void dfm_free(void *p) { free(p); }
}

#include "dfmetal_join.inc"
