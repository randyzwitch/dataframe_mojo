"""Radix-selected binary64 quantiles against independent host sorting."""
from pathlib import Path
import subprocess
import tempfile


def main():
    root = Path(__file__).parent
    def extract(name):
        return (root / name).read_text().split('R"metal(', 1)[1].split(')metal"', 1)[0]
    float64 = extract('dfmetal_float64.h')
    quantiles = extract('dfmetal_quantiles.h')
    source = r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <limits>
#include <vector>
using ulong=uint64_t;using uint=uint32_t;using uchar=uint8_t;
using std::min;using std::max;
#define device
#define thread
__FLOAT64__
__QUANTILES__
#undef device
#undef thread
struct Input { ulong x[64];uchar v[64];ulong q,index;uint begin,end,kind; };
void emulated(const Input &r,ulong *out){bool valid=false;
 out[0]=quantile_reduce(r.x,r.v,0UL,r.index,r.begin,r.end,r.q,r.kind,valid);out[1]=valid;}
int main(){@autoreleasepool{
 std::mt19937_64 random(0x5155414e54494c45UL);std::vector<Input> input;
 std::vector<ulong> boundary={0,0x8000000000000000UL,1,0x8000000000000001UL,0x0010000000000000UL,
  0x3ff0000000000000UL,0x4340000000000000UL,0x7fefffffffffffffUL,0x7ff0000000000000UL,
  0xfff0000000000000UL,0x7ff8000000000042UL,0xfff0000000000042UL};
 std::vector<double> probabilities={0.0,0.5,1.0,0.25,0.75,0.12345678901234567,
  std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::min(),std::nextafter(0.5,0.0),std::nextafter(0.5,1.0),std::nextafter(1.0,0.0)};
 for(uint i=0;i<5000;++i){Input r={};r.begin=random()%5;r.end=r.begin+random()%(33-r.begin);
  r.index=i%2?32UL:~0UL;r.kind=221+i%5;
  double q=i%3?probabilities[i%probabilities.size()]:double(random()>>11)*0x1p-53;r.q=std::bit_cast<ulong>(q);
  for(uint j=0;j<32;++j){r.x[j]=i<1000?boundary[(i+j)%boundary.size()]:i%3?random():std::bit_cast<ulong>(double(int(random()%101)-50));
    r.v[j]=random()%5!=0;r.x[32+j]=31-j;}
  input.push_back(r);}
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();if(!device){puts("SKIP quantiles: no Metal device");return 0;}
 NSString *shader=@R"shader(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
__FLOAT64__
__QUANTILES__
struct Input { ulong x[64];uchar v[64];ulong q,index;uint begin,end,kind; };
kernel void quantiles(device const Input *inputs[[buffer(0)]],device ulong *outputs[[buffer(1)]],uint i[[thread_position_in_grid]]){
 device const Input &r=inputs[i];bool valid=false;
 outputs[2*i]=quantile_reduce(r.x,r.v,0UL,r.index,r.begin,r.end,r.q,r.kind,valid);outputs[2*i+1]=ulong(valid);
}
)shader";
 NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
 id<MTLLibrary> library=[device newLibraryWithSource:shader options:options error:&error];
 if(!library){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
 id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"quantiles"] error:&error];
 if(!pipeline){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
 id<MTLBuffer> data=[device newBufferWithBytes:input.data() length:input.size()*sizeof(Input) options:MTLResourceStorageModeShared];
 id<MTLBuffer> output=[device newBufferWithLength:input.size()*2*sizeof(ulong) options:MTLResourceStorageModeShared];
 id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
 id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];
 [encoder setBuffer:data offset:0 atIndex:0];[encoder setBuffer:output offset:0 atIndex:1];
 [encoder dispatchThreads:MTLSizeMake(input.size(),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
 [encoder endEncoding];[command commit];[command waitUntilCompleted];
 if(command.status==MTLCommandBufferStatusError){fprintf(stderr,"%s\n",command.error.description.UTF8String);return 2;}
 const ulong *gpu=static_cast<const ulong*>(output.contents);
 for(size_t i=0;i<input.size();++i){const auto &r=input[i];std::vector<double> sorted;
  for(uint at=r.begin;at<r.end;++at){uint row=r.index==~0UL?at:uint(r.x[r.index+at]);if(r.v[row])sorted.push_back(std::bit_cast<double>(r.x[row]));}
  std::sort(sorted.begin(),sorted.end(),[](double a,double b){return !std::isnan(a)&&(std::isnan(b)||a<b);});
  double expected=0;bool valid=!sorted.empty();
  if(valid){double position=std::bit_cast<double>(r.q)*double(sorted.size()-1),fraction=position-std::floor(position);
   size_t lo=size_t(std::floor(position)),hi=std::min(size_t(std::ceil(position)),sorted.size()-1);
   if(r.kind==222)expected=sorted[lo];else if(r.kind==223)expected=sorted[hi];
   else if(r.kind==224)expected=sorted[std::min(size_t(std::floor(position+0.5)),sorted.size()-1)];
   else if(lo==hi||fraction==0)expected=sorted[lo];else if(r.kind==225)expected=(sorted[lo]+sorted[hi])/2;
   else expected=std::fma(sorted[hi]-sorted[lo],fraction,sorted[lo]);}
  ulong host[2];emulated(r,host);auto equal=[&](ulong word){double got=std::bit_cast<double>(word);return word==std::bit_cast<ulong>(expected)||(got==0&&expected==0)||(std::isnan(got)&&std::isnan(expected));};
  if(host[1]!=valid||gpu[2*i+1]!=valid||!equal(host[0])||!equal(gpu[2*i])){
   fprintf(stderr,"Mismatch case=%zu kind=%u q=%016llx n=%zu cpu=%016llx host=%016llx gpu=%016llx valid=%d/%llu/%llu\n",i,r.kind,(unsigned long long)r.q,sorted.size(),
    (unsigned long long)std::bit_cast<ulong>(expected),(unsigned long long)host[0],(unsigned long long)gpu[2*i],valid,(unsigned long long)host[1],(unsigned long long)gpu[2*i+1]);return 1;}}
 printf("PASS %zu quantile host/GPU sorting reference cases on %s\n",input.size(),device.name.UTF8String);return 0;
}}
'''.replace('__FLOAT64__', float64).replace('__QUANTILES__', quantiles)
    with tempfile.TemporaryDirectory(prefix='dfmetal-quantiles-') as directory:
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
