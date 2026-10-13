"""Compare decimal-to-Float64 conversion with separately rounded CPU arithmetic.

Run on Apple Silicon: python3 native/dfmetal/test_decimal_cast.py
Both the shared C++ primitives and their actual Metal execution must match
integer-to-Float64 conversion followed by Float64 division, bit for bit.
"""
from pathlib import Path
import subprocess
import tempfile


def main():
    header = Path(__file__).with_name('dfmetal_float64.h').read_text()
    primitives = header.split('R"metal(', 1)[1].split(')metal"', 1)[0]
    source = r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
using ulong = uint64_t;
using uint = uint32_t;
__FLOAT64_SOURCE__
struct Input { ulong low, high, divisor; };
int main() { @autoreleasepool {
  std::vector<Input> inputs;
  auto append = [&](unsigned __int128 magnitude, uint scale, bool negative) {
    unsigned __int128 bits = negative ? -magnitude : magnitude;
    unsigned __int128 unit = 1;
    for (uint j=0; j<scale; ++j) unit *= 10;
    inputs.push_back({ulong(bits), ulong(bits >> 64), std::bit_cast<ulong>(double(unit))});
  };
  unsigned __int128 maximum = 1;
  for (uint j=0; j<38; ++j) maximum *= 10;
  --maximum;
  for (uint scale=0; scale<=38; ++scale) {
    append(0,scale,false); append(maximum,scale,false); append(maximum,scale,true);
    for (uint exponent=0; exponent<127; ++exponent) {
      unsigned __int128 base = static_cast<unsigned __int128>(1) << exponent;
      for (int offset=-2; offset<=2; ++offset) {
        if (offset<0 && base<ulong(-offset)) continue;
        unsigned __int128 magnitude = offset<0 ? base-ulong(-offset) : base+ulong(offset);
        append(magnitude,scale,false); append(magnitude,scale,true);
      }
      if (exponent>=54) {
        unsigned __int128 half = base + (static_cast<unsigned __int128>(1) << (exponent-53));
        for (int offset=-2; offset<=2; ++offset) {
          unsigned __int128 magnitude = offset<0 ? half-ulong(-offset) : half+ulong(offset);
          append(magnitude,scale,false); append(magnitude,scale,true);
        }
      }
    }
  }
  std::mt19937_64 random(0x4341535444454349UL);
  for (size_t i=0; i<200000; ++i) {
    unsigned __int128 magnitude = (static_cast<unsigned __int128>(random() & 0x3fffffffffffffffUL) << 64) | random();
    append(magnitude,uint(random()%39),(random()&1)!=0);
  }
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) { puts("SKIP decimal cast GPU differential tests: no Metal device"); return 0; }
  NSString *shader = @R"shader(
#include <metal_stdlib>
using namespace metal;
__FLOAT64_SOURCE__
struct Input { ulong low, high, divisor; };
kernel void casts(device const Input *input[[buffer(0)]],
                  device ulong *out[[buffer(1)]], uint i[[thread_position_in_grid]]) {
  Input row=input[i];
  out[i]=f64_div(f64_from_wide(row.low,row.high,false),row.divisor);
}
)shader";
  NSError *error=nil;
  MTLCompileOptions *options=[MTLCompileOptions new];
  options.mathMode=MTLMathModeSafe;
  id<MTLLibrary> library=[device newLibraryWithSource:shader options:options error:&error];
  if (!library) { fprintf(stderr,"%s\n",error.description.UTF8String); return 2; }
  id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"casts"] error:&error];
  if (!pipeline) { fprintf(stderr,"%s\n",error.description.UTF8String); return 2; }
  id<MTLCommandQueue> queue=[device newCommandQueue];
  id<MTLBuffer> data=[device newBufferWithBytes:inputs.data() length:inputs.size()*sizeof(Input) options:MTLResourceStorageModeShared];
  id<MTLBuffer> result=[device newBufferWithLength:inputs.size()*sizeof(ulong) options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> command=[queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline]; [encoder setBuffer:data offset:0 atIndex:0]; [encoder setBuffer:result offset:0 atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(inputs.size(),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
  if (command.status==MTLCommandBufferStatusError) { fprintf(stderr,"%s\n",command.error.description.UTF8String); return 2; }
  auto gpu=static_cast<const ulong *>(result.contents);
  for (size_t i=0; i<inputs.size(); ++i) {
    auto row=inputs[i];
    unsigned __int128 bits=(static_cast<unsigned __int128>(row.high) << 64) | row.low;
    __int128 value=std::bit_cast<__int128>(bits);
    volatile double converted=double(value);
    double expected_value=converted / std::bit_cast<double>(row.divisor);
    ulong expected=std::bit_cast<ulong>(expected_value);
    ulong host=f64_div(f64_from_wide(row.low,row.high,false),row.divisor);
    if (expected!=host || expected!=gpu[i]) {
      fprintf(stderr,"Mismatch i=%zu low=%016llx high=%016llx divisor=%016llx expected=%016llx host=%016llx gpu=%016llx\n",i,
              (unsigned long long)row.low,(unsigned long long)row.high,(unsigned long long)row.divisor,
              (unsigned long long)expected,(unsigned long long)host,(unsigned long long)gpu[i]);return 1;
    }
  }
  printf("PASS %zu exact decimal cast host/GPU differential cases on %s\n",inputs.size(),device.name.UTF8String);
  return 0;
}}
'''.replace('__FLOAT64_SOURCE__', primitives)
    with tempfile.TemporaryDirectory(prefix='dfmetal-decimal-cast-') as directory:
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
