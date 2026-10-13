"""Compare shared decimal cast primitives against independent host arithmetic."""
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
struct Input { ulong low,high,fromScale,toScale,precision,bits,isUnsigned,isBoolean; };
Wide power(uint scale) { Wide unit=1; for(uint i=0;i<scale;++i)unit*=10; return unit; }
int main() { @autoreleasepool {
  std::vector<Input> inputs;
  auto append=[&](Wide raw,uint from,uint to,uint precision,uint bits,bool uns,bool boolean) {
    inputs.push_back({ulong(raw),ulong(raw>>64),from,to,precision,bits,uns,boolean});
  };
  for(uint from=0;from<=38;++from)for(uint to=0;to<=38;++to) {
    for(Wide raw : {Wide(0),Wide(1),Wide(5),Wide(15),Wide(25),Wide(35),power(38)-1,
                    (Wide(1)<<63)-1,Wide(1)<<63,(Wide(1)<<64)-1,Wide(1)<<100}) {
      for(bool negative : {false,true}) {
        Wide word=negative?-raw:raw;
        append(word,from,to,38,128,false,false);
        if(!to)for(uint bits : {8U,16U,32U,64U})for(bool uns : {false,true})
          append(word,from,0,0,bits,uns,false);
      }
    }
  }
  std::mt19937_64 random(0x44454343415354UL);
  for(uint i=0;i<250000;++i) {
    Wide raw=(Wide(random())<<64)|random();
    raw%=power(38);
    if(random()&1)raw=-raw;
    uint precision=uint(random()%39),from=uint(random()%39);
    uint to=precision?uint(random()%(precision+1)):0;
    uint bits=precision?128:8U<<(random()%4);
    append(raw,from,to,precision,bits,random()&1,!precision&&(random()%5==0));
  }
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();
  if(!device){puts("SKIP decimal casts: no Metal device");return 0;}
  NSString *shader=@R"shader(
#include <metal_stdlib>
using namespace metal;
__FLOAT64_SOURCE__
__DECIMAL_SOURCE__
struct Input { ulong low,high,fromScale,toScale,precision,bits,isUnsigned,isBoolean; };
kernel void casts(device const Input *input[[buffer(0)]],device ulong *out[[buffer(1)]],
                  uint i[[thread_position_in_grid]]) {
  Input r=input[i];
  DecimalCast value=decimal_cast(ulong2(r.low,r.high),uint(r.fromScale),uint(r.toScale),
      uint(r.precision),uint(r.bits),r.isUnsigned!=0,r.isBoolean!=0);
  out[3*i]=value.value.x;out[3*i+1]=value.value.y;out[3*i+2]=ulong(value.valid);
}
)shader";
  NSError *error=nil;
  MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
  id<MTLLibrary> library=[device newLibraryWithSource:shader options:options error:&error];
  if(!library){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
  id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:
      [library newFunctionWithName:@"casts"] error:&error];
  if(!pipeline){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
  id<MTLCommandQueue> queue=[device newCommandQueue];
  id<MTLBuffer> data=[device newBufferWithBytes:inputs.data() length:inputs.size()*sizeof(Input)
      options:MTLResourceStorageModeShared];
  id<MTLBuffer> output=[device newBufferWithLength:inputs.size()*3*sizeof(ulong)
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
    auto r=inputs[i];Wide word=(Wide(r.high)<<64)|r.low;
    if(r.fromScale>r.toScale) {
      bool negative=word>>127;
      Wide magnitude=negative?-word:word,unit=power(uint(r.fromScale-r.toScale));
      Wide q=magnitude/unit,remainder=magnitude%unit;
      if(remainder>unit-remainder||(remainder==unit-remainder&&(q&1)))++q;
      word=negative?-q:q;
    } else word*=power(uint(r.toScale-r.fromScale));
    bool negative=word>>127,valid;
    Wide magnitude=negative?-word:word;
    if(r.precision)valid=magnitude<power(uint(r.precision));
    else if(r.isBoolean){word=magnitude!=0;valid=true;}
    else {
      Wide maximum=r.isUnsigned?(Wide(1)<<r.bits)-1:(Wide(1)<<(r.bits-1))-!negative;
      valid=magnitude<=maximum&&!(r.isUnsigned&&negative);
    }
    DecimalCast host=decimal_cast(ulong2(r.low,r.high),uint(r.fromScale),uint(r.toScale),
        uint(r.precision),uint(r.bits),r.isUnsigned!=0,r.isBoolean!=0);
    if(host.valid!=valid||gpu[3*i+2]!=ulong(valid)||
       (valid&&(host.value.x!=ulong(word)||host.value.y!=ulong(word>>64)||
                gpu[3*i]!=ulong(word)||gpu[3*i+1]!=ulong(word>>64)))) {
      fprintf(stderr,"Mismatch i=%zu from=%llu to=%llu precision=%llu bits=%llu raw=%016llx:%016llx\n",
          i,(unsigned long long)r.fromScale,(unsigned long long)r.toScale,
          (unsigned long long)r.precision,(unsigned long long)r.bits,
          (unsigned long long)r.high,(unsigned long long)r.low);return 1;
    }
  }
  printf("PASS %zu decimal cast host/GPU differential cases on %s\n",inputs.size(),device.name.UTF8String);
  return 0;
}}
'''.replace('__FLOAT64_SOURCE__', extract(root / 'dfmetal_float64.h')).replace('__DECIMAL_SOURCE__', extract(root / 'dfmetal_decimal.h'))
    with tempfile.TemporaryDirectory(prefix='dfmetal-decimal-casts-') as directory:
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
