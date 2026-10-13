"""Compare shared integer-only exp/log against independent host libm on Metal."""
from pathlib import Path
import subprocess
import tempfile


def main():
    header = Path(__file__).with_name("dfmetal_float64.h").read_text()
    primitives = header.split('R"metal(', 1)[1].split(')metal"', 1)[0]
    source = r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
using ulong=uint64_t; using uint=uint32_t;
PRIMITIVES
int main(){@autoreleasepool {
  std::vector<ulong> values={0,1,2,0x000fffffffffffffUL,0x0010000000000000UL,
    0x3fefffffffffffffUL,0x3ff0000000000000UL,0x3ff0000000000001UL,
    0x7fefffffffffffffUL,0x7ff0000000000000UL,0x7ff0000000000042UL,
    0x7ff8000000000042UL};
  auto initial=values.size();for(size_t i=0;i<initial;++i)values.push_back(values[i]|(1UL<<63));
  for(double value:{-750.,-745.1332191019411,-744.4400719213812,-710.,-1.,1.,
                    1.4142135623730951,709.782712893384,710.}){
    for(int delta=-16;delta<=16;++delta)values.push_back(std::bit_cast<ulong>(value)+delta);
  }
  for(ulong exponent=1;exponent<2047;++exponent){
    for(ulong fraction:{0UL,1UL,0x000fffffffffffffUL})values.push_back((exponent<<52)|fraction);
  }
  std::mt19937_64 random(0xE7A109);
  for(size_t i=0;i<25000;++i){
    values.push_back(random());
    values.push_back(std::bit_cast<ulong>(std::uniform_real_distribution<double>(-750.,710.)(random)));
  }
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();if(!device)return 2;
  NSString *shader=@R"shader(
#include <metal_stdlib>
using namespace metal;
PRIMITIVES
kernel void apply(device const ulong *x[[buffer(0)]],device ulong *y[[buffer(1)]],
 constant uint2 &shape[[buffer(2)]],uint row[[thread_position_in_grid]]){
 if(row>=shape.x)return;y[row]=shape.y?f64_log(x[row]):f64_exp(x[row]);
}
)shader";
  NSError *error=nil;
  MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
  id<MTLLibrary> library=[device newLibraryWithSource:shader options:options error:&error];
  if(!library){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 3;}
  id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"apply"] error:&error];
  if(!pipeline)return 4;
  auto input=[device newBufferWithBytes:values.data() length:values.size()*8 options:MTLResourceStorageModeShared];
  auto output=[device newBufferWithLength:values.size()*8 options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> queue=[device newCommandQueue];
  for(uint op=0;op<2;++op){
    uint shape[2]={uint(values.size()),op};
    auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];[encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];[encoder setBytes:shape length:8 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(values.size(),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    if(command.error){fprintf(stderr,"%s\n",command.error.localizedDescription.UTF8String);return 5;}
    auto actual=static_cast<ulong *>(output.contents);ulong maximum=0;
    for(size_t i=0;i<values.size();++i){
      ulong host=op?f64_log(values[i]):f64_exp(values[i]);
      if(actual[i]!=host){printf("shared GPU/host mismatch op=%u input=%016llx actual=%016llx host=%016llx\n",op,values[i],actual[i],host);return 6;}
      double x=std::bit_cast<double>(values[i]),expected=op?std::log(x):std::exp(x);
      ulong reference=std::bit_cast<ulong>(expected);
      if(std::isnan(expected)){if(!std::isnan(std::bit_cast<double>(actual[i])))return 7;continue;}
      if(!std::isfinite(expected)||expected==0){if(actual[i]!=reference){printf("class mismatch op=%u x=%.17g actual=%016llx ref=%016llx\n",op,x,actual[i],reference);return 8;}continue;}
      ulong distance=actual[i]>reference?actual[i]-reference:reference-actual[i];
      if(distance>maximum)maximum=distance;
      if(distance>16){printf("accuracy mismatch op=%u x=%.17g actual=%.17g ref=%.17g ULP=%llu\n",op,x,std::bit_cast<double>(actual[i]),expected,distance);return 9;}
    }
    printf("PASS %s: %zu GPU/host encodings, maximum %llu ULP vs independent libm\n",op?"log":"exp",values.size(),maximum);
  }
  if(f64_exp(0x7ff0000000000042UL)!=0x7ff8000000000042UL || f64_log(0xfff0000000000042UL)!=0xfff8000000000042UL)return 10;
  return 0;
}}
'''.replace("PRIMITIVES", primitives)
    with tempfile.TemporaryDirectory(prefix="dfmetal-transcendentals-") as directory:
        path = Path(directory)
        (path / "test.mm").write_text(source)
        subprocess.run([
            "xcrun", "clang++", "-O2", "-ffp-contract=off", "-std=c++20", "-fobjc-arc",
            str(path / "test.mm"), "-framework", "Metal", "-framework", "Foundation",
            "-o", str(path / "test"),
        ], check=True)
        subprocess.run([str(path / "test")], check=True)


if __name__ == "__main__":
    main()
