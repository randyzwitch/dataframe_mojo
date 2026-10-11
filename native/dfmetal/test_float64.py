"""Compare integer-only binary64 primitives with host IEEE arithmetic on Metal.

Run on Apple Silicon: python3 native/dfmetal/test_float64.py
The exact same source is compiled as both C++ and Metal. All finite results,
infinities, and signed zeros must match bit for bit. NaNs must remain NaNs;
payload propagation is checked separately because native hosts differ there.
"""

from pathlib import Path
import subprocess
import tempfile


def main():
    header = Path(__file__).with_name("dfmetal_float64.h").read_text()
    primitives = header.split('R"metal(', 1)[1].split(')metal"', 1)[0]
    integer_helper = r'''
inline ulong test_integer(ulong a, uint op) {
  uint pair=(op-13)/2;
  F64Integer converted=f64_to_integer(a,pair<2?64:8,(pair&1)!=0);
  return (op&1) ? (converted.valid?converted.word:0UL) : ulong(converted.valid);
}
'''
    source = r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
using ulong = uint64_t;
using uint = uint32_t;
PRIMITIVES
INTEGER_TEST_HELPER
int main() { @autoreleasepool {
  std::vector<ulong> boundary = {
    0, 1, 2, 3, 0x0007ffffffffffffUL, 0x000ffffffffffffeUL,
    0x000fffffffffffffUL, 0x0010000000000000UL, 0x0010000000000001UL,
    0x3c90000000000000UL, 0x3ca0000000000000UL, 0x3cb0000000000000UL,
    0x3fd0000000000000UL, 0x3fd5555555555555UL, 0x3fefffffffffffffUL,
    0x3ff0000000000000UL, 0x3ff0000000000001UL, 0x3ff0000000000002UL,
    0x4000000000000000UL, 0x4340000000000000UL, 0x7fdfffffffffffffUL,
    0x7feffffffffffffeUL, 0x7fefffffffffffffUL, 0x7ff0000000000000UL,
    0x7ff0000000000001UL, 0x7ff8000000000042UL, 0x7fffffffffffffffUL,
    0x43dfffffffffffffUL, 0x43e0000000000000UL, 0x43e0000000000001UL,
    0x43efffffffffffffUL, 0x43f0000000000000UL,
    0x368fffffffffffffUL, 0x3690000000000000UL, 0x3690000000000001UL,
    0x36a0000000000000UL, 0x47efffffe0000000UL,
    0x47efffffefffffffUL, 0x47effffff0000000UL, 0x47effffff0000001UL
  };
  auto count = boundary.size();
  for (size_t i = 0; i < count; ++i) boundary.push_back(boundary[i] | (1UL << 63));
  std::vector<ulong> left, right;
  for (auto a : boundary) for (auto b : boundary) {
    left.push_back(a); right.push_back(b);
  }
  std::mt19937_64 random(0x464c4f41543634UL);
  for (size_t i = 0; i < 250000; ++i) {
    left.push_back(random()); right.push_back(random());
  }
  // Near-cancellation and rounding ties deserve focused coverage in addition
  // to uniformly distributed encodings (whose exponents are often far apart).
  for (size_t i = 0; i < 25000; ++i) {
    ulong a = random();
    left.push_back(a);
    right.push_back((a ^ (1UL << 63)) + (i % 9) - 4);
  }
  if (f64_add(0x7ff0000000000042UL, 0) != 0x7ff8000000000042UL ||
      f64_sub(0, 0xfff0000000000042UL) != 0xfff8000000000042UL)
    return 2;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) { puts("SKIP binary64 GPU differential tests: no Metal device"); return 0; }
  NSString *shader = @R"shader(
#include <metal_stdlib>
using namespace metal;
PRIMITIVES
INTEGER_TEST_HELPER
kernel void arithmetic(device const ulong *a[[buffer(0)]],
 device const ulong *b[[buffer(1)]],device ulong *out[[buffer(2)]],
 constant uint &op[[buffer(3)]],uint i[[thread_position_in_grid]]) {
 out[i] = op==0 ? f64_add(a[i],b[i]) : op==1 ? f64_sub(a[i],b[i]) :
          op==2 ? f64_mul(a[i],b[i]) : op==3 ? f64_div(a[i],b[i]) :
          op==4 ? ulong(f64_equal(a[i],b[i])) : op==5 ? ulong(f64_less(a[i],b[i])) :
          op==6 ? f64_from_unsigned(a[i]) : op==7 ? f64_from_signed(as_type<long>(a[i])) :
          op==8 ? f64_from_float32(uint(a[i])) : op==9 ? ulong(f64_to_float32(a[i])) :
          op==10 ? f64_integral(a[i],55) : op==11 ? f64_integral(a[i],56) :
          op==12 ? f64_round(a[i]) : test_integer(a[i],op);
}
)shader";
  NSError *error = nil;
  MTLCompileOptions *options = [MTLCompileOptions new];
  options.mathMode = MTLMathModeSafe;
  id<MTLLibrary> library = [device newLibraryWithSource:shader options:options error:&error];
  if (!library) { fprintf(stderr, "%s\n", error.description.UTF8String); return 2; }
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:
    [library newFunctionWithName:@"arithmetic"] error:&error];
  if (!pipeline) { fprintf(stderr, "%s\n", error.description.UTF8String); return 2; }
  id<MTLCommandQueue> queue = [device newCommandQueue];
  size_t bytes = left.size() * sizeof(ulong);
  id<MTLBuffer> a = [device newBufferWithBytes:left.data() length:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> b = [device newBufferWithBytes:right.data() length:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> out = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  size_t checked = 0;
  for (uint op = 0; op < 21; ++op) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a offset:0 atIndex:0]; [encoder setBuffer:b offset:0 atIndex:1];
    [encoder setBuffer:out offset:0 atIndex:2]; [encoder setBytes:&op length:sizeof(op) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(left.size(),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
      fprintf(stderr, "%s\n", command.error.description.UTF8String); return 2;
    }
    auto gpu = static_cast<const ulong *>(out.contents);
    for (size_t i = 0; i < left.size(); ++i) {
      volatile double x = std::bit_cast<double>(left[i]), y = std::bit_cast<double>(right[i]);
      double value = op==0 ? x+y : op==1 ? x-y : op==2 ? x*y : op==3 ? x/y : 0;
      ulong expected = op==4 ? ulong(x==y) : op==5 ? ulong(x<y) : std::bit_cast<ulong>(value);
      if (op==6) expected=std::bit_cast<ulong>(double(left[i]));
      if (op==7) expected=std::bit_cast<ulong>(double(std::bit_cast<int64_t>(left[i])));
      if (op==8) expected=std::bit_cast<ulong>(double(std::bit_cast<float>(uint(left[i]))));
      if (op==9) expected=std::bit_cast<uint>(float(x));
      if (op==10) expected=std::bit_cast<ulong>(std::floor(x));
      if (op==11) expected=std::bit_cast<ulong>(std::ceil(x));
      if (op==12) {
        double rounded=std::isnan(x)||std::isinf(x)||std::abs(x)>=4503599627370496.0 ? x : std::floor(std::abs(x)+0.5);
        if (!(std::isnan(x)||std::isinf(x)||std::abs(x)>=4503599627370496.0) && x<0) rounded=-rounded;
        expected=std::bit_cast<ulong>(rounded);
      }
      if (op>=13) {
        uint pair=(op-13)/2, width=pair<2?64:8;
        bool uns=(pair&1)!=0;
        double whole=std::trunc(x), upper=std::ldexp(1.0,int(width)-(uns?0:1));
        bool valid=std::isfinite(x)&&whole<upper&&whole>=(uns?0.0:-upper);
        expected=(op&1) ? (valid?(uns?ulong(whole):std::bit_cast<ulong>(int64_t(whole))):0UL) : ulong(valid);
      }
      ulong host = op==0 ? f64_add(left[i],right[i]) : op==1 ? f64_sub(left[i],right[i]) :
                   op==2 ? f64_mul(left[i],right[i]) : op==3 ? f64_div(left[i],right[i]) :
                   op==4 ? ulong(f64_equal(left[i],right[i])) : op==5 ? ulong(f64_less(left[i],right[i])) :
                   op==6 ? f64_from_unsigned(left[i]) : op==7 ? f64_from_signed(std::bit_cast<int64_t>(left[i])) :
                   op==8 ? f64_from_float32(uint(left[i])) : op==9 ? ulong(f64_to_float32(left[i])) :
                   op==10 ? f64_integral(left[i],55) : op==11 ? f64_integral(left[i],56) :
                   op==12 ? f64_round(left[i]) : test_integer(left[i],op);
      bool nan = op>=13 ? false : op==9 ? (expected&0x7fffffffUL)>0x7f800000UL : f64_nan(expected);
      bool actualNan = op==9 ? (host&0x7fffffffUL)>0x7f800000UL && (gpu[i]&0x7fffffffUL)>0x7f800000UL : f64_nan(host) && f64_nan(gpu[i]);
      if ((nan ? !actualNan : host != expected || gpu[i] != expected)) {
        fprintf(stderr, "Mismatch op=%u a=%016llx b=%016llx expected=%016llx host=%016llx gpu=%016llx\n",
          op, (unsigned long long)left[i], (unsigned long long)right[i],
          (unsigned long long)expected, (unsigned long long)host, (unsigned long long)gpu[i]);
        return 1;
      }
      ++checked;
    }
  }
  printf("PASS %zu binary64 host/GPU differential cases on %s\n", checked, device.name.UTF8String);
  return 0;
}}
'''.replace("PRIMITIVES", primitives).replace("INTEGER_TEST_HELPER", integer_helper)
    with tempfile.TemporaryDirectory(prefix="dfmetal-float64-") as directory:
        root = Path(directory)
        program = root / "test.mm"
        binary = root / "test"
        program.write_text(source)
        subprocess.run(
            ["xcrun", "clang++", "-O2", "-std=c++20", "-fobjc-arc",
             "-ffp-contract=off", "-fno-fast-math", "-mmacosx-version-min=15.0", str(program),
             "-framework", "Metal", "-framework", "Foundation", "-o", str(binary)],
            check=True,
        )
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
