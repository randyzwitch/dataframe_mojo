"""Compare shared decimal multiplication/division primitives against independent host arithmetic."""
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
#include <utility>
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
struct Input { ulong low,high,rightLow,rightHigh,leftScale,rightScale,op,bits; };
Wide power(uint scale) { Wide unit=1; for(uint i=0;i<scale;++i)unit*=10; return unit; }

using Signed = __int128;
struct Model { Wide value; uint error; };
Wide magnitude(Wide word) { return word>>127?-word:word; }
bool precision(Wide word) { return magnitude(word)>=power(38); }
std::pair<Wide,Wide> divideSigned(Wide a,Wide b) {
  Signed x=static_cast<Signed>(a),y=static_cast<Signed>(b);
  Signed quotient=x/y,remainder=x%y;
  if(remainder&&((remainder<0)!=(y<0)))remainder+=y;
  return {Wide(quotient),Wide(remainder)};
}
Wide halfEven(Wide word,uint scale) {
  bool negative=word>>127;
  Signed m=static_cast<Signed>(negative?-word:word),unit=static_cast<Signed>(power(scale));
  if(m<=INT64_MAX&&unit<=INT64_MAX)m=int64_t(m);
  Signed q=m/unit,r=m%unit;
  if(r<0){--q;r+=unit;}
  Signed rest=unit-r;
  if(r>rest||(r==rest&&(q&1)))++q;
  return negative?-Wide(q):Wide(q);
}
Model model(Input input) {
  Wide a=(Wide(input.high)<<64)|input.low,b=(Wide(input.rightHigh)<<64)|input.rightLow;
  uint common=uint(input.leftScale>input.rightScale?input.leftScale:input.rightScale);
  Wide maximum=(Wide(1)<<127)-1;
  if((common>input.leftScale&&magnitude(a)>maximum/power(common-uint(input.leftScale)))||
     (common>input.rightScale&&magnitude(b)>maximum/power(common-uint(input.rightScale))))
    return {0,1};
  if(input.op==7) {
    if(a==(Wide(1)<<127))return {0,1};
    if(!a)return {0,0};
    Wide bound=b>>127?Wide(1)<<127:maximum;
    if(magnitude(b)>bound/magnitude(a))return {0,1};
    Wide value=a*b;
    uint scale=uint(input.leftScale<input.rightScale?input.leftScale:input.rightScale);
    if(scale)value=halfEven(value,scale);
    return {value,precision(value)?2U:0U};
  }
  if(!b)return {0,4};
  bool negative=(a>>127)!=(b>>127);
  Wide numerator=magnitude(a),denominator=magnitude(b);
  auto parts=divideSigned(numerator,denominator);
  Wide value=parts.first,remainder=parts.second;
  if(precision(value))return {0,2};
  uint shift=common+uint(input.rightScale)-uint(input.leftScale);
  for(uint i=0;i<shift;++i) {
    value*=10;
    if(precision(value))return {0,2};
    remainder*=10;
    parts=divideSigned(remainder,denominator);
    value+=parts.first;remainder=parts.second;
  }
  Signed rest=static_cast<Signed>(denominator-remainder),r=static_cast<Signed>(remainder);
  if(r>rest||(r==rest&&(value&1)))++value;
  if(precision(value))return {0,2};
  return {negative?-value:value,0};
}
int main() { @autoreleasepool {
  std::vector<Input> inputs;
  auto append=[&](Wide a,Wide b,uint left,uint right) {
    for(uint op : {7U,24U})
      inputs.push_back({ulong(a),ulong(a>>64),ulong(b),ulong(b>>64),left,right,op,128});
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
  for(uint scale=0;scale<=38;++scale) {
    for(Wide a : {Wide(0),Wide(1),Wide(2),Wide(3),Wide(25),Wide(75),power(38)-1,Wide(1)<<126,Wide(1)<<127})
      for(Wide b : {Wide(0),Wide(1),Wide(2),Wide(3),Wide(2)<<64,power(38)-1,Wide(1)<<126,Wide(1)<<127})
        for(bool na : {false,true})for(bool nb : {false,true})
          append(na?-a:a,nb?-b:b,scale,scale);
  }
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();
  if(!device){puts("SKIP decimal multiplication/division: no Metal device");return 0;}
  NSString *shader=@R"shader(
#include <metal_stdlib>
using namespace metal;
__FLOAT64_SOURCE__
__DECIMAL_SOURCE__
struct Input { ulong low,high,rightLow,rightHigh,leftScale,rightScale,op,bits; };
kernel void comparisons(device const Input *input[[buffer(0)]],device ulong *out[[buffer(1)]],
                        uint i[[thread_position_in_grid]]) {
  Input r=input[i];
  DecimalArithmetic value=r.op==7
      ?decimal_multiply(ulong2(r.low,r.high),ulong2(r.rightLow,r.rightHigh),uint(r.leftScale),uint(r.rightScale))
      :decimal_divide_rows(ulong2(r.low,r.high),ulong2(r.rightLow,r.rightHigh),uint(r.leftScale),uint(r.rightScale));
  out[3*i]=value.value.x;out[3*i+1]=value.value.y;out[3*i+2]=ulong(value.error);
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
    auto r=inputs[i];
    Wide a=(Wide(r.high)<<64)|r.low,b=(Wide(r.rightHigh)<<64)|r.rightLow;
    Model expected=model(r);
    DecimalArithmetic host=r.op==7
        ?decimal_multiply(ulong2(r.low,r.high),ulong2(r.rightLow,r.rightHigh),uint(r.leftScale),uint(r.rightScale))
        :decimal_divide_rows(ulong2(r.low,r.high),ulong2(r.rightLow,r.rightHigh),uint(r.leftScale),uint(r.rightScale));
    if(host.error!=expected.error||gpu[3*i+2]!=ulong(expected.error)||
       (!expected.error&&(host.value.x!=ulong(expected.value)||host.value.y!=ulong(expected.value>>64)||
                 gpu[3*i]!=ulong(expected.value)||gpu[3*i+1]!=ulong(expected.value>>64)))) {
      fprintf(stderr,"Mismatch i=%zu leftScale=%llu rightScale=%llu op=%llu expected_error=%u host_error=%u gpu_error=%llu\n",i,
          (unsigned long long)r.leftScale,(unsigned long long)r.rightScale,
          (unsigned long long)r.op,expected.error,host.error,(unsigned long long)gpu[3*i+2]);return 1;
    }
  }
  printf("PASS %zu decimal multiplication/division host/GPU differential cases on %s\n",inputs.size(),device.name.UTF8String);
  return 0;
}}
'''.replace('__FLOAT64_SOURCE__', extract(root / 'dfmetal_float64.h')).replace('__DECIMAL_SOURCE__', extract(root / 'dfmetal_decimal.h'))
    with tempfile.TemporaryDirectory(prefix='dfmetal-decimal-multiplicative-') as directory:
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
