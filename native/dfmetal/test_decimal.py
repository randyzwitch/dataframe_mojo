"""Compare the shared Decimal32/64 mean with the CPU quotient/remainder algorithm.

Run on Apple Silicon: python3 native/dfmetal/test_decimal.py
Every result must match bit for bit in both C++ and Metal. Native descriptors
validate widths and scales separately; these cases exercise rounding and signs.
"""
from pathlib import Path
import subprocess
import tempfile


def main():
    root = Path(__file__).parent
    extract = lambda p: p.read_text().split('R"metal(', 1)[1].split(')metal"', 1)[0]
    source = r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <random>
#include <initializer_list>
#include <vector>
using ulong = uint64_t;
using uint = uint32_t;
struct ulong2 {
  ulong x, y;
  ulong2(ulong all) : x(all), y(all) {}
  ulong2(ulong low, ulong high) : x(low), y(high) {}
};
inline ulong mulhi(ulong a, ulong b) {
  return ulong((static_cast<unsigned __int128>(a) * b) >> 64);
}
__FLOAT64_SOURCE__
__DECIMAL_SOURCE__
struct Input { ulong low, high, count, scale; };
int main() { @autoreleasepool {
  std::vector<Input> inputs;
  auto append = [&](unsigned __int128 magnitude, ulong count, uint scale, bool negative) {
    unsigned __int128 bits = negative ? -magnitude : magnitude;
    inputs.push_back({ulong(bits), ulong(bits >> 64), count, scale});
  };
  for (uint scale = 0; scale <= 18; ++scale) {
    for (ulong count : {1UL, 2UL, 3UL, 7UL, 31UL, 257UL, 65537UL, 2147483647UL}) {
      for (ulong whole : {0UL, 1UL, 2UL, 3UL, (1UL<<53)-1, 1UL<<53, (1UL<<53)+1,
                         999999999999999999UL, (1UL<<63)-1, 1UL<<63}) {
        for (ulong remainder : std::initializer_list<ulong>{0UL, count / 2, count - 1}) {
          unsigned __int128 magnitude = static_cast<unsigned __int128>(whole) * count + remainder;
          if (magnitude > static_cast<unsigned __int128>(count) * (1UL<<63)) continue;
          append(magnitude, count, scale, false);
          if (magnitude) append(magnitude, count, scale, true);
        }
      }
    }
  }
  std::mt19937_64 random(0x444543494d414cUL);
  for (size_t i = 0; i < 200000; ++i) {
    ulong count = random() % 2147483647UL + 1;
    ulong whole = random() & ((1UL<<63)-1);
    unsigned __int128 magnitude = static_cast<unsigned __int128>(whole) * count + random() % count;
    append(magnitude, count, uint(random() % 19), (random() & 1) != 0);
  }
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) { puts("SKIP decimal mean GPU differential tests: no Metal device"); return 0; }
  NSString *shader = @R"shader(
#include <metal_stdlib>
using namespace metal;
__FLOAT64_SOURCE__
__DECIMAL_SOURCE__
struct Input { ulong low, high, count, scale; };
kernel void means(device const Input *input[[buffer(0)]],
                  device ulong *out[[buffer(1)]], uint i[[thread_position_in_grid]]) {
  Input row = input[i];
  out[i] = f64_decimal_mean(row.low, row.high, row.count, uint(row.scale));
}
)shader";
  NSError *error = nil;
  MTLCompileOptions *options = [MTLCompileOptions new];
  options.mathMode = MTLMathModeSafe;
  id<MTLLibrary> library = [device newLibraryWithSource:shader options:options error:&error];
  if (!library) { fprintf(stderr, "%s\n", error.description.UTF8String); return 2; }
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:
    [library newFunctionWithName:@"means"] error:&error];
  if (!pipeline) { fprintf(stderr, "%s\n", error.description.UTF8String); return 2; }
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLBuffer> data = [device newBufferWithBytes:inputs.data() length:inputs.size()*sizeof(Input) options:MTLResourceStorageModeShared];
  id<MTLBuffer> result = [device newBufferWithLength:inputs.size()*sizeof(ulong) options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:data offset:0 atIndex:0]; [encoder setBuffer:result offset:0 atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(inputs.size(),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
  if (command.status == MTLCommandBufferStatusError) {
    fprintf(stderr, "%s\n", command.error.description.UTF8String); return 2;
  }
  auto gpu = static_cast<const ulong *>(result.contents);
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto row = inputs[i];
    bool negative = row.high >> 63;
    unsigned __int128 bits = (static_cast<unsigned __int128>(row.high) << 64) | row.low;
    unsigned __int128 magnitude = negative ? -bits : bits;
    unsigned __int128 per_row = magnitude / row.count;
    unsigned __int128 left = magnitude - per_row * row.count;
    unsigned __int128 unit = 1;
    for (uint j = 0; j < row.scale; ++j) unit *= 10;
    unsigned __int128 whole = per_row / unit;
    unsigned __int128 digits = per_row - whole * unit;
    // The separate roundings are part of the CPU decimal_mean contract.
    volatile double fraction = double(left) / double(row.count);
    volatile double with_digits = double(digits) + fraction;
    volatile double scaled = with_digits / double(unit);
    double value = double(whole) + scaled;
    if (negative) value = -value;
    ulong expected = std::bit_cast<ulong>(value);
    ulong host = f64_decimal_mean(row.low, row.high, row.count, uint(row.scale));
    if (expected != host || expected != gpu[i]) {
      fprintf(stderr, "Mismatch i=%zu low=%016llx high=%016llx count=%llu scale=%llu expected=%016llx host=%016llx gpu=%016llx\n",
              i, (unsigned long long)row.low, (unsigned long long)row.high,
              (unsigned long long)row.count, (unsigned long long)row.scale,
              (unsigned long long)expected, (unsigned long long)host, (unsigned long long)gpu[i]);
      return 1;
    }
  }
  printf("PASS %zu exact decimal mean host/GPU differential cases on %s\n", inputs.size(), device.name.UTF8String);
  return 0;
}}
'''.replace('__FLOAT64_SOURCE__', extract(root / 'dfmetal_float64.h')).replace('__DECIMAL_SOURCE__', extract(root / 'dfmetal_decimal.h'))
    with tempfile.TemporaryDirectory(prefix='dfmetal-decimal-') as directory:
        program = Path(directory) / 'test.mm'
        binary = Path(directory) / 'test'
        program.write_text(source)
        subprocess.run([
            'xcrun', 'clang++', '-O2', '-std=c++20', '-fobjc-arc',
            '-ffp-contract=off', '-fno-fast-math', '-mmacosx-version-min=15.0',
            str(program), '-framework', 'Metal', '-framework', 'Foundation',
            '-o', str(binary),
        ], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
