"""Compare shared decimal comparison primitives against independent host arithmetic."""
from pathlib import Path
import subprocess
import tempfile


def main():
    root = Path(__file__).parent
    extract = lambda p: p.read_text().split('R"metal(', 1)[1].split(')metal"', 1)[0]
    source = r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
using ulong = uint64_t;
using uint = uint32_t;
using Wide = unsigned __int128;
struct ulong2 {
  ulong x,y;
  ulong2(ulong all) : x(all),y(all) {}
  ulong2(ulong low,ulong high) : x(low),y(high) {}
};
inline ulong mulhi(ulong a,ulong b) { return ulong((Wide(a)*b)>>64); }
__FLOAT64_SOURCE__
__DECIMAL_SOURCE__
struct Input { ulong low,high,rightLow,rightHigh,leftScale,rightScale,op; };
Wide power(uint scale) { Wide unit=1; for(uint i=0;i<scale;++i)unit*=10; return unit; }
int main() { @autoreleasepool {
  std::vector<Input> inputs;
  auto append=[&](Wide a,Wide b,uint left,uint right) {
    for(uint op : {8U,9U,20U,21U,22U,23U})
      inputs.push_back({ulong(a),ulong(a>>64),ulong(b),ulong(b>>64),left,right,op});
  };
  Wide maximum=(Wide(1)<<127)-1;
  for(uint left=0;left<=38;++left)for(uint right=0;right<=38;++right) {
    uint delta=left>right?left-right:right-left;
    Wide boundary=maximum/power(delta);
    for(Wide raw : {Wide(0),Wide(1),power(38)-1,Wide(1)<<63,
                   (Wide(1)<<64)-1,Wide(1)<<100,boundary,boundary+1,maximum})
      for(bool negative : {false,true}) {
        Wide word=negative?-raw:raw;
        append(word,word,left,right);
        append(word,-word,left,right);
      }
  }
  std::mt19937_64 random(0x444543434f4d50UL);
  for(uint i=0;i<50000;++i) {
    Wide a=((Wide(random())<<64)|random())%power(38);
    Wide b=((Wide(random())<<64)|random())%power(38);
    if(random()&1)a=-a;
    if(random()&1)b=-b;
    append(a,b,uint(random()%39),uint(random()%39));
  }
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();
  if(!device){puts("SKIP decimal comparisons: no Metal device");return 0;}
  NSString *shader=@R"shader(
#include <metal_stdlib>
using namespace metal;
__FLOAT64_SOURCE__
__DECIMAL_SOURCE__
struct Input { ulong low,high,rightLow,rightHigh,leftScale,rightScale,op; };
kernel void comparisons(device const Input *input[[buffer(0)]],device ulong *out[[buffer(1)]],
                        uint i[[thread_position_in_grid]]) {
  Input r=input[i];
  DecimalComparison value=decimal_compare(ulong2(r.low,r.high),ulong2(r.rightLow,r.rightHigh),
      uint(r.leftScale),uint(r.rightScale),uint(r.op));
  out[2*i]=ulong(value.value);out[2*i+1]=ulong(value.valid);
}
)shader";
  NSError *error=nil;
  MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
  id<MTLLibrary> library=[device newLibraryWithSource:shader options:options error:&error];
  if(!library){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
  id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:
      [library newFunctionWithName:@"comparisons"] error:&error];
  if(!pipeline){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
  id<MTLCommandQueue> queue=[device newCommandQueue];
  id<MTLBuffer> data=[device newBufferWithBytes:inputs.data() length:inputs.size()*sizeof(Input)
      options:MTLResourceStorageModeShared];
  id<MTLBuffer> output=[device newBufferWithLength:inputs.size()*2*sizeof(ulong)
      options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> command=[queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];[encoder setBuffer:data offset:0 atIndex:0];
  [encoder setBuffer:output offset:0 atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(inputs.size(),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [encoder endEncoding];[command commit];[command waitUntilCompleted];
  if(command.status==MTLCommandBufferStatusError){fprintf(stderr,"%s\n",command.error.description.UTF8String);return 2;}
  auto gpu=static_cast<const ulong*>(output.contents);
  for(size_t i=0;i<inputs.size();++i) {
    auto r=inputs[i];
    Wide a=(Wide(r.high)<<64)|r.low,b=(Wide(r.rightHigh)<<64)|r.rightLow;
    bool valid=true;
    auto scale=[&](Wide word,uint delta) {
      Wide factor=power(delta),magnitude=(word>>127)?-word:word;
      if(delta&&magnitude>maximum/factor)valid=false;
      return word*factor;
    };
    uint common=uint(r.leftScale>r.rightScale?r.leftScale:r.rightScale);
    a=scale(a,common-uint(r.leftScale));b=scale(b,common-uint(r.rightScale));
    __int128 sa=static_cast<__int128>(a),sb=static_cast<__int128>(b);
    bool answer=r.op==8?sa>sb:r.op==9?sa==sb:r.op==20?sa<sb:
                r.op==21?sa>=sb:r.op==22?sa<=sb:sa!=sb;
    DecimalComparison host=decimal_compare(ulong2(r.low,r.high),ulong2(r.rightLow,r.rightHigh),
        uint(r.leftScale),uint(r.rightScale),uint(r.op));
    if(host.valid!=valid||gpu[2*i+1]!=ulong(valid)||
       (valid&&(host.value!=answer||gpu[2*i]!=ulong(answer)))) {
      fprintf(stderr,"Mismatch i=%zu leftScale=%llu rightScale=%llu op=%llu\n",i,
          (unsigned long long)r.leftScale,(unsigned long long)r.rightScale,
          (unsigned long long)r.op);return 1;
    }
  }
  printf("PASS %zu decimal comparison host/GPU differential cases on %s\n",inputs.size(),device.name.UTF8String);
  return 0;
}}
'''.replace('__FLOAT64_SOURCE__', extract(root / 'dfmetal_float64.h')).replace('__DECIMAL_SOURCE__', extract(root / 'dfmetal_decimal.h'))
    with tempfile.TemporaryDirectory(prefix='dfmetal-decimal-comparisons-') as directory:
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
