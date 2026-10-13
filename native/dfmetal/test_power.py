"""Validate integer-only Metal floating power against libm and Decimal boundaries."""
from pathlib import Path
import subprocess,tempfile,struct,math
from decimal import Decimal,localcontext
p=Path(__file__).with_name('dfmetal_float64.h').read_text().split('R"metal(',1)[1].split(')metal"',1)[0]
dense=[]
for a in [math.nextafter(1.,math.inf), math.nextafter(1.,0.),1.01,1.1,3.,10.,.1,1e-300,1e300]:
 for threshold in [709.782712893384,-745.1332191019411,-744.4400719213812]:
  center=struct.unpack('Q',struct.pack('d',threshold/math.log(a)))[0]
  for delta in range(-128,129):
   b=struct.unpack('d',struct.pack('Q',center+delta))[0]
   with localcontext() as context:
    context.prec=160
    value=float((Decimal.from_float(a).ln()*Decimal.from_float(b)).exp())
   fields=[struct.unpack('Q',struct.pack('d',x))[0] for x in [a,b,value]]
   dense.append('{%dUL,%dUL,0UL,%dUL,1UL}'%tuple(fields))
s=r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
using ulong=uint64_t;using uint=uint32_t;
PRIMITIVES
struct Input {ulong a,b,type,reference,has_reference;};
int main(){@autoreleasepool{
 std::vector<Input> input;std::mt19937_64 random(0x504f57);
 for(ulong type:{0UL,1UL}) {
  auto add=[&](double a,double b){input.push_back(type?Input{std::bit_cast<uint>(float(a)),std::bit_cast<uint>(float(b)),type}:Input{std::bit_cast<ulong>(a),std::bit_cast<ulong>(b),type});};
  std::vector<double> edges={0.,-0.,1.,-1.,2.,-2.,.5,-.5,INFINITY,-INFINITY,NAN,0x1p-1074,0x1p-1022,0x1.fffffffffffffp1023,0x1.0000000000001p0,0x1.fffffffffffffp-1,0x1p-149,0x1p-126,0x1.fffffep127};
  for(auto a:edges)for(auto b:edges)add(a,b);
  for(auto a:edges)for(int b:{-1075,-1074,-1024,-150,-149,-127,-126,-65,-64,-3,-2,-1,0,1,2,3,64,65,127,128,1023,1024,1074})add(a,b);
  for(int i=0;i<15000;++i){
   double a=std::bit_cast<double>(random()),b=std::bit_cast<double>(random());
   switch(i%5){case 1:a=std::uniform_real_distribution<double>(.001,1000)(random);b=std::uniform_real_distribution<double>(-100,100)(random);break;
    case 2:a=std::bit_cast<double>(0x3ff0000000000000UL+(random()%2049)-1024);b=std::uniform_real_distribution<double>(-1e18,1e18)(random);break;
    case 3:a=std::uniform_real_distribution<double>(-100,100)(random);b=double(int(random()%129)-64);break;
    case 4:a=std::ldexp(1.,int(random()%2098)-1074);b=double(int(random()%2098)-1074);break;}
   add(a,b);
  }
 }
 __DENSE__
 auto device=MTLCreateSystemDefaultDevice();NSError *error=nil;MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
 NSString *shader=@R"shader(
#include <metal_stdlib>
using namespace metal;
PRIMITIVES
struct Input{ulong a,b,type,reference,has_reference;};
kernel void apply(device const Input *x[[buffer(0)]],device ulong *y[[buffer(1)]],constant uint &n[[buffer(2)]],uint row[[thread_position_in_grid]]){
 if(row>=n)return;Input r=x[row];ulong a=r.type?f64_from_float32(uint(r.a)):r.a,b=r.type?f64_from_float32(uint(r.b)):r.b;
 ulong out=f64_power(a,b);y[row]=r.type?ulong(f64_to_float32(out)):out;
}
)shader";
 auto library=[device newLibraryWithSource:shader options:options error:&error];if(!library){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 2;}
 auto pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"apply"] error:&error];if(!pipeline)return 3;
 auto x=[device newBufferWithBytes:input.data() length:input.size()*sizeof(Input) options:MTLResourceStorageModeShared];auto y=[device newBufferWithLength:input.size()*8 options:MTLResourceStorageModeShared];
 uint n=uint(input.size());auto command=[[device newCommandQueue] commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];[encoder setBuffer:x offset:0 atIndex:0];[encoder setBuffer:y offset:0 atIndex:1];[encoder setBytes:&n length:4 atIndex:2];[encoder dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];if(command.error){fprintf(stderr,"%s\n",command.error.localizedDescription.UTF8String);return 4;}
 auto result=(ulong*)y.contents;double maxRelative[2]={};ulong maxULP[2]={};
 for(uint i=0;i<n;++i){auto r=input[i];ulong a=r.type?f64_from_float32(uint(r.a)):r.a,b=r.type?f64_from_float32(uint(r.b)):r.b;
  ulong emulated=f64_power(a,b);if(r.type)emulated=f64_to_float32(emulated);if(emulated!=result[i]){printf("host GPU mismatch %u\n",i);return 5;}
  double aa=std::bit_cast<double>(a),bb=std::bit_cast<double>(b);double reference=r.has_reference?std::bit_cast<double>(r.reference):r.type?double(std::pow(float(aa),float(bb))):std::pow(aa,bb);
  double actual=r.type?double(std::bit_cast<float>(uint(result[i]))):std::bit_cast<double>(result[i]);
  if(std::isnan(reference)){if(!std::isnan(actual)){printf("NaN %u\n",i);return 6;}continue;}
  if(!std::isfinite(reference)||!reference){if(reference!=actual||std::signbit(reference)!=std::signbit(actual)){printf("CLASS %u %.17g ** %.17g actual %.17g ref %.17g\n",i,aa,bb,actual,reference);return 7;}continue;}
  double rel=std::abs((actual-reference)/reference);maxRelative[r.type]=std::max(maxRelative[r.type],rel);
  ulong ref=r.type?ulong(std::bit_cast<uint>(float(reference))):std::bit_cast<ulong>(reference);
  ulong distance=result[i]>ref?result[i]-ref:ref-result[i];maxULP[r.type]=std::max(maxULP[r.type],distance);
  if(distance>16){printf("ULP %u %.17g ** %.17g distance %llu\n",i,aa,bb,static_cast<unsigned long long>(distance));return 9;}
  double tolerance=r.type?1e-5:5e-12,absolute=r.type?1e-44:1e-323;
  if(!std::isfinite(actual)||std::abs(actual-reference)>tolerance*std::abs(reference)+absolute){printf("ACCURACY %u %.17g ** %.17g actual %.17g ref %.17g\n",i,aa,bb,actual,reference);return 8;}
 }
 printf("PASS %u GPU power cases host bits exact max relative F64 %.17g F32 %.17g max ULP F64 %llu F32 %llu\n",n,maxRelative[0],maxRelative[1],static_cast<unsigned long long>(maxULP[0]),static_cast<unsigned long long>(maxULP[1]));
}}
'''.replace('PRIMITIVES',p).replace('__DENSE__','const Input boundaries[]={'+','.join(dense)+'};input.insert(input.end(),boundaries,boundaries+sizeof(boundaries)/sizeof(Input));')
with tempfile.TemporaryDirectory(prefix='dfmetal-power-prototype-') as d:
 f=Path(d)/'test.mm';f.write_text(s);binary=Path(d)/'test'
 subprocess.run(['xcrun','clang++','-O2','-ffp-contract=off','-std=c++20','-fobjc-arc',str(f),'-framework','Metal','-framework','Foundation','-o',str(binary)],check=True)
 subprocess.run([str(binary)],check=True)
