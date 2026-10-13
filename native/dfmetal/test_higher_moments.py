"""Binary64 skewness/kurtosis states against an independent host model."""
from pathlib import Path
import subprocess
import tempfile


def main():
    root = Path(__file__).parent
    extract = lambda name: (root / name).read_text().split('R"metal(', 1)[1].split(')metal"', 1)[0]
    body = r'''
  CentralMomentState a={0,0,0,0,0},b={0,0,0,0,0};
  for(uint j=0;j<16;++j)if(r.valid&(1UL<<j)){
    if(j<r.split)a=central_add(a,r.x[j]);else b=central_add(b,r.x[j]);}
  a=central_merge(a,b);out[0]=a.count;out[1]=a.mean;out[2]=a.m2;out[3]=a.m3;out[4]=a.m4;
  ulong mask=0;
  for(uint j=0;j<6;++j){bool kurtosis=j>=2;ulong flags=j<2?1UL-j:(j==2?3UL:j==3?1UL:j==4?2UL:0UL);
    bool valid=central_valid(a.count,kurtosis,flags);mask|=ulong(valid)<<j;
    out[5+j]=valid?central_finish(a,kurtosis,flags):0UL;}
  out[11]=mask;
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
struct Input { ulong x[16],valid,split; };
struct HostMoments {
  ulong count=0;double mean=0,m2=0,m3=0,m4=0;
  void add(double value){double before=double(count++),n=double(count),d=value-mean,dn=d/n,dn2=dn*dn,t=d*dn*before;
    mean+=dn;m4+=t*dn2*(n*n-3*n+3)+6*dn2*m2-4*dn*m3;m3+=t*dn*(n-2)-3*dn*m2;m2+=t;}
  void merge(HostMoments o){if(!o.count)return;if(!count){*this=o;return;}
    double a=double(count),b=double(o.count),n=a+b,d=o.mean-mean,d2=d*d;
    double second=m2+o.m2+d2*a*b/n;
    double third=m3+o.m3+d2*d*a*b*(a-b)/(n*n)+3*d*(a*o.m2-b*m2)/n;
    double fourth=m4+o.m4+d2*d2*a*b*(a*a-a*b+b*b)/(n*n*n)+6*d2*(a*a*o.m2+b*b*m2)/(n*n)+4*d*(a*o.m3-b*m3)/n;
    mean+=d*b/n;m2=second;m3=third;m4=fourth;count+=o.count;}
  bool valid(bool kurtosis,ulong flags){return count&&((flags&1)||count>=(kurtosis?4UL:3UL));}
  double finish(bool kurtosis,ulong flags){double n=double(count);
    if(!kurtosis){double g=std::sqrt(n)*m3/(m2*std::sqrt(m2));return flags&1?g:g*std::sqrt(n*(n-1))/(n-2);}
    double e=n*m4/(m2*m2)-3;if(!(flags&1))e=((n+1)*e+6)*(n-1)/((n-2)*(n-3));return flags&2?e:e+3;}
};
void emulated(Input r,ulong *out){__BODY__}
int main(){@autoreleasepool{
  std::mt19937_64 random(0x484947484d4f4dUL);std::vector<Input> input;
  std::vector<ulong> boundary={0,0x8000000000000000UL,1,0x8000000000000001UL,0x0010000000000000UL,
    0x3ff0000000000000UL,0x4340000000000000UL,0x7fefffffffffffffUL,0x7ff0000000000000UL,
    0xfff0000000000000UL,0x7ff8000000000042UL,0xfff0000000000042UL};
  for(uint i=0;i<30000;++i){Input r={};r.valid=random()&65535;r.split=random()%17;
    for(uint j=0;j<16;++j){if(i<1000)r.x[j]=boundary[(i+j)%boundary.size()];
      else if(i%3==0)r.x[j]=random();else r.x[j]=std::bit_cast<ulong>((i%2?1e12:0.0)+double(int(random()%101)-50)*0.1);}
    input.push_back(r);}
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();if(!device){puts("SKIP higher moments: no Metal device");return 0;}
  NSString *shader=@R"shader(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
__FLOAT64__
__MOMENTS__
struct Input { ulong x[16],valid,split; };
kernel void moments(device const Input *inputs[[buffer(0)]],device ulong *outputs[[buffer(1)]],uint i[[thread_position_in_grid]]){
  Input r=inputs[i];device ulong *out=outputs+i*12;__BODY__
}
)shader";
  NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
  id<MTLLibrary> library=[device newLibraryWithSource:shader options:options error:&error];
  if(!library){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
  id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"moments"] error:&error];
  if(!pipeline){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
  id<MTLBuffer> data=[device newBufferWithBytes:input.data() length:input.size()*sizeof(Input) options:MTLResourceStorageModeShared];
  id<MTLBuffer> output=[device newBufferWithLength:input.size()*12*sizeof(ulong) options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];
  [encoder setBuffer:data offset:0 atIndex:0];[encoder setBuffer:output offset:0 atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(input.size(),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [encoder endEncoding];[command commit];[command waitUntilCompleted];
  if(command.status==MTLCommandBufferStatusError){fprintf(stderr,"%s\n",command.error.description.UTF8String);return 2;}
  const ulong *gpu=static_cast<const ulong*>(output.contents);size_t cases=0;
  for(size_t i=0;i<input.size();++i){auto r=input[i];HostMoments a,b;
    for(uint j=0;j<16;++j)if(r.valid&(1UL<<j)){double x=std::bit_cast<double>(r.x[j]);if(j<r.split)a.add(x);else b.add(x);}
    a.merge(b);ulong expected[12]={a.count,std::bit_cast<ulong>(a.mean),std::bit_cast<ulong>(a.m2),std::bit_cast<ulong>(a.m3),std::bit_cast<ulong>(a.m4)};
    for(uint j=0;j<6;++j){bool kurtosis=j>=2;ulong flags=j<2?1UL-j:(j==2?3UL:j==3?1UL:j==4?2UL:0UL);
      bool valid=a.valid(kurtosis,flags);expected[11]|=ulong(valid)<<j;expected[5+j]=valid?std::bit_cast<ulong>(a.finish(kurtosis,flags)):0UL;}
    ulong host[12];emulated(r,host);
    for(uint j=0;j<12;++j){auto equal=[&](ulong x,ulong y){return x==y||(j!=0&&j!=11&&f64_nan(x)&&f64_nan(y));};
      if(!equal(host[j],expected[j])||!equal(gpu[12*i+j],expected[j])){
        fprintf(stderr,"Mismatch case=%zu field=%u cpu=%016llx host=%016llx gpu=%016llx\n",i,j,
          (unsigned long long)expected[j],(unsigned long long)host[j],(unsigned long long)gpu[12*i+j]);return 1;}++cases;}}
  printf("PASS %zu higher-moment host/GPU differential fields on %s\n",cases,device.name.UTF8String);return 0;
}}
'''.replace('__BODY__', body).replace('__FLOAT64__', extract('dfmetal_float64.h')).replace('__MOMENTS__', extract('dfmetal_moments.h'))
    with tempfile.TemporaryDirectory(prefix='dfmetal-higher-moments-') as directory:
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
