"""Compare Welford/Chan variance and covariance helpers with host IEEE arithmetic."""
from pathlib import Path
import subprocess
import tempfile


def main():
    root = Path(__file__).parent
    extract = lambda path: path.read_text().split('R"metal(', 1)[1].split(')metal"', 1)[0]
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
struct Input { ulong x[12],y[12],valid,split,ddof; };
struct HostVariance {
  ulong count=0;double mean=0,m2=0;
  void add(double value){++count;double delta=value-mean;mean+=delta/double(count);m2+=delta*(value-mean);}
  void merge(HostVariance b){if(!b.count)return;if(!count){*this=b;return;}
    ulong total=count+b.count;double delta=b.mean-mean,weight=double(b.count)/double(total);
    mean+=delta*weight;m2+=b.m2+delta*delta*double(count)*double(b.count)/double(total);count=total;}
};
struct HostCovariance {
  ulong count=0;double meanX=0,meanY=0,cxy=0,m2x=0,m2y=0;
  void add(double x,double y){++count;double n=double(count),dx=x-meanX;meanX+=dx/n;
    double dy=y-meanY;meanY+=dy/n;cxy+=dx*(y-meanY);m2x+=dx*(x-meanX);m2y+=dy*(y-meanY);}
  void merge(HostCovariance o){if(!o.count)return;if(!count){*this=o;return;}
    double a=double(count),b=double(o.count),n=a+b,dx=o.meanX-meanX,dy=o.meanY-meanY;
    cxy+=o.cxy+dx*dy*a*b/n;m2x+=o.m2x+dx*dx*a*b/n;m2y+=o.m2y+dy*dy*a*b/n;
    meanX+=dx*b/n;meanY+=dy*b/n;count+=o.count;}
};
void emulated(Input r,ulong *out){
  VarianceState v={0,0,0},v2={0,0,0};CovarianceState c={0,0,0,0,0,0},c2={0,0,0,0,0,0};
  for(uint j=0;j<12;++j)if(r.valid&(1UL<<j)){
    if(j<r.split){v=variance_add(v,r.x[j]);c=covariance_add(c,r.x[j],r.y[j]);}
    else{v2=variance_add(v2,r.x[j]);c2=covariance_add(c2,r.x[j],r.y[j]);}}
  v=variance_merge(v,v2);c=covariance_merge(c,c2);
  out[0]=v.count;out[1]=v.mean;out[2]=v.m2;out[3]=c.count;out[4]=c.meanX;out[5]=c.meanY;
  out[6]=c.cxy;out[7]=c.m2x;out[8]=c.m2y;
  bool valid=v.count>r.ddof;
  out[9]=valid?variance_finish(v,r.ddof,false):0;
  out[10]=valid?variance_finish(v,r.ddof,true):0;
  out[11]=c.count?covariance_finish(c,r.ddof,false):0;
  out[12]=covariance_finish(c,r.ddof,true);out[13]=ulong(valid)|ulong(c.count!=0)<<1;
}
int main(){@autoreleasepool{
  std::vector<Input> input;std::mt19937_64 random(0x4d4f4d454e5453UL);
  std::vector<ulong> boundary={0,0x8000000000000000UL,1,0x8000000000000001UL,
    0x0010000000000000UL,0x4340000000000000UL,0x3ff0000000000000UL,
    0x7fefffffffffffffUL,0x7ff0000000000000UL,0xfff0000000000000UL,
    0x7ff8000000000042UL,0xfff0000000000042UL};
  for(uint i=0;i<30000;++i){Input r={};r.valid=random()&4095;r.split=random()%13;r.ddof=random()%15;
    if(i%31==0)r.ddof=0x7fffffffffffffffUL;
    for(uint j=0;j<12;++j){
      if(i<1000){r.x[j]=boundary[(i+j)%boundary.size()];r.y[j]=boundary[(i*3+j)%boundary.size()];}
      else if(i%3==0){r.x[j]=random();r.y[j]=random();}
      else {double base=i%2?1e12:0.0;r.x[j]=std::bit_cast<ulong>(base+double(int(random()%101)-50)*0.1);
        r.y[j]=std::bit_cast<ulong>(base+double(int(random()%101)-50)*0.2);}}
    input.push_back(r);}
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();if(!device){puts("SKIP moments: no Metal device");return 0;}
  NSString *shader=@R"shader(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
__FLOAT64__
__MOMENTS__
struct Input { ulong x[12],y[12],valid,split,ddof; };
kernel void moments(device const Input *input[[buffer(0)]],device ulong *out[[buffer(1)]],uint i[[thread_position_in_grid]]){
  Input r=input[i];VarianceState v={0,0,0},v2={0,0,0};CovarianceState c={0,0,0,0,0,0},c2={0,0,0,0,0,0};
  for(uint j=0;j<12;++j)if(r.valid&(1UL<<j)){
    if(j<r.split){v=variance_add(v,r.x[j]);c=covariance_add(c,r.x[j],r.y[j]);}
    else{v2=variance_add(v2,r.x[j]);c2=covariance_add(c2,r.x[j],r.y[j]);}}
  v=variance_merge(v,v2);c=covariance_merge(c,c2);uint at=i*14;
  out[at]=v.count;out[at+1]=v.mean;out[at+2]=v.m2;out[at+3]=c.count;out[at+4]=c.meanX;out[at+5]=c.meanY;
  out[at+6]=c.cxy;out[at+7]=c.m2x;out[at+8]=c.m2y;bool valid=v.count>r.ddof;
  out[at+9]=valid?variance_finish(v,r.ddof,false):0;out[at+10]=valid?variance_finish(v,r.ddof,true):0;
  out[at+11]=c.count?covariance_finish(c,r.ddof,false):0;
  out[at+12]=covariance_finish(c,r.ddof,true);out[at+13]=ulong(valid)|ulong(c.count!=0)<<1;
}
)shader";
  NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
  id<MTLLibrary> library=[device newLibraryWithSource:shader options:options error:&error];
  if(!library){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
  id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"moments"] error:&error];
  if(!pipeline){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
  id<MTLBuffer> data=[device newBufferWithBytes:input.data() length:input.size()*sizeof(Input) options:MTLResourceStorageModeShared];
  id<MTLBuffer> output=[device newBufferWithLength:input.size()*14*sizeof(ulong) options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];
  [encoder setBuffer:data offset:0 atIndex:0];[encoder setBuffer:output offset:0 atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(input.size(),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [encoder endEncoding];[command commit];[command waitUntilCompleted];
  if(command.status==MTLCommandBufferStatusError){fprintf(stderr,"%s\n",command.error.description.UTF8String);return 2;}
  const ulong *gpu=static_cast<const ulong*>(output.contents);size_t cases=0;
  for(size_t i=0;i<input.size();++i){auto r=input[i];HostVariance v,v2;HostCovariance c,c2;
    for(uint j=0;j<12;++j)if(r.valid&(1UL<<j)){double x=std::bit_cast<double>(r.x[j]),y=std::bit_cast<double>(r.y[j]);
      if(j<r.split){v.add(x);c.add(x,y);}else{v2.add(x);c2.add(x,y);}}
    v.merge(v2);c.merge(c2);bool valid=v.count>r.ddof;
    double variance=valid?v.m2/double(v.count-r.ddof):0;
    double covariance=c.count?(c.count<=r.ddof?0:c.cxy/double(c.count-r.ddof)):0;
    double correlation=c.count<2?0.0/0.0:c.cxy/std::sqrt(c.m2x*c.m2y);
    ulong reference[14]={v.count,std::bit_cast<ulong>(v.mean),std::bit_cast<ulong>(v.m2),c.count,
      std::bit_cast<ulong>(c.meanX),std::bit_cast<ulong>(c.meanY),std::bit_cast<ulong>(c.cxy),
      std::bit_cast<ulong>(c.m2x),std::bit_cast<ulong>(c.m2y),std::bit_cast<ulong>(variance),
      std::bit_cast<ulong>(std::sqrt(variance)),std::bit_cast<ulong>(covariance),std::bit_cast<ulong>(correlation),
      ulong(valid)|ulong(c.count!=0)<<1};
    ulong host[14];emulated(r,host);
    for(uint j=0;j<14;++j){bool numeric=j!=0&&j!=3&&j!=13;
      auto equal=[&](ulong a,ulong b){return a==b||(numeric&&f64_nan(a)&&f64_nan(b));};
      if(!equal(host[j],reference[j])||!equal(gpu[14*i+j],reference[j])){
        fprintf(stderr,"Mismatch case=%zu field=%u cpu=%016llx host=%016llx gpu=%016llx\n",i,j,
          (unsigned long long)reference[j],(unsigned long long)host[j],(unsigned long long)gpu[14*i+j]);return 1;}++cases;}}
  printf("PASS %zu variance/covariance host/GPU differential fields on %s\n",cases,device.name.UTF8String);return 0;
}}
'''.replace('__FLOAT64__', extract(root / 'dfmetal_float64.h')).replace('__MOMENTS__', extract(root / 'dfmetal_moments.h'))
    with tempfile.TemporaryDirectory(prefix='dfmetal-moments-') as directory:
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
