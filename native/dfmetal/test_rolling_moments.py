"""Compare sliding moment state words against independent host IEEE arithmetic."""
from pathlib import Path
import subprocess
import tempfile


def main():
    root = Path(__file__).parent
    extract = lambda path: path.read_text().split('R"metal(', 1)[1].split(')metal"', 1)[0]
    declarations = r'''
struct Input { ulong x[32],valid,window,ddof,minimum; };
'''
    emulated = r'''
void emulated(Input r,ulong *out){
  ulong origin=0;
  for(uint j=0;j<32;++j)if((r.valid&(1UL<<j))&&f64_abs(r.x[j])<0x7ff0000000000000UL){origin=r.x[j];break;}
  uint lo=0,hi=0;RollingVarianceState state={0,0,0,0};
  for(uint j=0;j<32;++j){uint start=j+1>r.window?j+1-uint(r.window):0;
    if(start<lo||start>=hi){lo=start;hi=start;state={0,0,0,0};}
    while(lo<start){uint at=lo++;if(r.valid&(1UL<<at))state=rolling_variance_remove(state,r.x[at],origin);}
    while(hi<=j){uint at=hi++;if(r.valid&(1UL<<at))state=rolling_variance_add(state,r.x[at],origin);}
    ulong total=state.count+state.bad;bool valid=total>=r.minimum&&total>r.ddof;
    uint at=j*8;out[at]=state.count;out[at+1]=state.bad;out[at+2]=state.mean;out[at+3]=state.m2;
    out[at+4]=origin;out[at+5]=valid?rolling_variance_finish(state,r.ddof,false):0;
    out[at+6]=valid?rolling_variance_finish(state,r.ddof,true):0;out[at+7]=valid;
  }
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
using ulong=uint64_t;using uint=uint32_t;
__FLOAT64__
__MOMENTS__
__DECLARATIONS__
__EMULATED__
void reference(Input r,ulong *out){
  double origin=0;
  for(uint j=0;j<32;++j)if((r.valid&(1UL<<j))&&std::isfinite(std::bit_cast<double>(r.x[j]))){origin=std::bit_cast<double>(r.x[j]);break;}
  uint lo=0,hi=0;ulong count=0,bad=0;double mean=0,m2=0;
  for(uint j=0;j<32;++j){uint start=j+1>r.window?j+1-uint(r.window):0;
    if(start<lo||start>=hi){lo=start;hi=start;count=bad=0;mean=m2=0;}
    while(lo<start){uint at=lo++;if(!(r.valid&(1UL<<at)))continue;
      double x=std::bit_cast<double>(r.x[at]);if(!std::isfinite(x)){--bad;continue;}
      x-=origin;--count;if(!count){mean=m2=0;}else{double delta=x-mean;mean-=delta/double(count);
        m2=std::fma(-delta,x-mean,m2);m2=std::fmax(m2,0.0);}}
    while(hi<=j){uint at=hi++;if(!(r.valid&(1UL<<at)))continue;
      double x=std::bit_cast<double>(r.x[at]);if(!std::isfinite(x)){++bad;continue;}
      x-=origin;++count;double delta=x-mean;mean+=delta/double(count);m2=std::fma(delta,x-mean,m2);}
    ulong total=count+bad;bool valid=total>=r.minimum&&total>r.ddof;
    double variance=valid?(bad?NAN:std::fmax(m2,0.0)/double(count-r.ddof)):0;
    uint at=j*8;out[at]=count;out[at+1]=bad;out[at+2]=std::bit_cast<ulong>(mean);out[at+3]=std::bit_cast<ulong>(m2);
    out[at+4]=std::bit_cast<ulong>(origin);out[at+5]=std::bit_cast<ulong>(variance);
    out[at+6]=std::bit_cast<ulong>(std::sqrt(variance));out[at+7]=valid;
  }
}
int main(){@autoreleasepool{
  std::vector<Input> input;std::mt19937_64 random(0x534c4944494e47UL);
  std::vector<ulong> boundary={0,0x8000000000000000UL,1,0x8000000000000001UL,
    0x0010000000000000UL,0x4340000000000000UL,0x3ff0000000000000UL,
    0x7fefffffffffffffUL,0xffefffffffffffffUL,0x7ff0000000000000UL,0xfff0000000000000UL,
    0x7ff8000000000042UL,0xfff0000000000042UL};
  for(uint i=0;i<12000;++i){Input r={};r.valid=random()&0xffffffffUL;r.window=1+random()%40;r.ddof=random()%5;r.minimum=random()%5;
    if(i%31==0)r.ddof=0xffffffffUL;if(i%17==0)r.window=0xffffffffUL;
    for(uint j=0;j<32;++j){
      if(i<1000)r.x[j]=boundary[(i+j)%boundary.size()];
      else if(i%3==0)r.x[j]=random();
      else {double base=i%2?1e12:0.0;r.x[j]=std::bit_cast<ulong>(base+double(int(random()%101)-50)*0.1);}}
    input.push_back(r);}
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();if(!device){puts("SKIP rolling moments: no Metal device");return 0;}
  NSString *shader=@R"shader(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
__FLOAT64__
__MOMENTS__
__DECLARATIONS__
__GPU_EMULATED__
kernel void moments(device const Input *input[[buffer(0)]],device ulong *out[[buffer(1)]],uint i[[thread_position_in_grid]]){
  emulated(input[i],out+ulong(i)*256);
}
)shader";
  NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
  id<MTLLibrary> library=[device newLibraryWithSource:shader options:options error:&error];
  if(!library){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
  id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"moments"] error:&error];
  if(!pipeline){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
  id<MTLBuffer> data=[device newBufferWithBytes:input.data() length:input.size()*sizeof(Input) options:MTLResourceStorageModeShared];
  id<MTLBuffer> output=[device newBufferWithLength:input.size()*256*sizeof(ulong) options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];
  [encoder setBuffer:data offset:0 atIndex:0];[encoder setBuffer:output offset:0 atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(input.size(),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [encoder endEncoding];[command commit];[command waitUntilCompleted];
  if(command.status==MTLCommandBufferStatusError){fprintf(stderr,"%s\n",command.error.description.UTF8String);return 2;}
  const ulong *gpu=static_cast<const ulong*>(output.contents);size_t cases=0;
  for(size_t i=0;i<input.size();++i){ulong host[256],ref[256];emulated(input[i],host);reference(input[i],ref);
    for(uint j=0;j<256;++j){bool numeric=j%8>=2&&j%8<=6;
      auto equal=[&](ulong a,ulong b){return a==b||(numeric&&f64_nan(a)&&f64_nan(b));};
      if(!equal(host[j],ref[j])||!equal(gpu[256*i+j],ref[j])){
        fprintf(stderr,"Mismatch case=%zu field=%u cpu=%016llx host=%016llx gpu=%016llx\n",i,j,
          (unsigned long long)ref[j],(unsigned long long)host[j],(unsigned long long)gpu[256*i+j]);return 1;}++cases;}}
  printf("PASS %zu sliding variance host/GPU differential fields on %s\n",cases,device.name.UTF8String);return 0;
}}
'''.replace('__FLOAT64__', extract(root / 'dfmetal_float64.h')).replace('__MOMENTS__', extract(root / 'dfmetal_moments.h')).replace('__DECLARATIONS__', declarations).replace('__EMULATED__', emulated).replace('__GPU_EMULATED__', emulated.replace('ulong *out', 'device ulong *out'))
    with tempfile.TemporaryDirectory(prefix='dfmetal-rolling-moments-') as directory:
        program = Path(directory) / 'test.mm'
        binary = Path(directory) / 'test'
        program.write_text(source)
        subprocess.run(['xcrun', 'clang++', '-O2', '-std=c++20', '-fobjc-arc',
                        '-ffp-contract=off', '-fno-fast-math', '-mmacosx-version-min=15.0',
                        str(program), '-framework', 'Metal', '-framework', 'Foundation',
                        '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
