"""Compare Mojo GPU floating primitives with an independent host IEEE oracle.

The C++ program only generates expected bits with host hardware/libm. It neither
loads Metal nor includes the production emulation code. Temporary files are
removed after the Mojo device test completes.
"""
from pathlib import Path
import subprocess
import tempfile


ORACLE = r'''
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
using U = uint64_t;
using V = uint32_t;
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  FILE *out = std::fopen(argv[1], "wb");
  if (!out) return 2;
  std::vector<U> boundary = {
    0, 1, 2, 3, 0x0007ffffffffffffULL, 0x000fffffffffffffULL,
    0x0010000000000000ULL, 0x0010000000000001ULL,
    0x3ca0000000000000ULL, 0x3fefffffffffffffULL, 0x3ff0000000000000ULL,
    0x3ff0000000000001ULL, 0x4000000000000000ULL, 0x4340000000000000ULL,
    0x7fefffffffffffffULL, 0x7ff0000000000000ULL, 0x7ff0000000000042ULL,
    0x7ff8000000000042ULL, 0x43dfffffffffffffULL, 0x43e0000000000000ULL,
    0x43efffffffffffffULL, 0x43f0000000000000ULL,
    0x3690000000000000ULL, 0x47effffff0000000ULL,
    0x007fffffULL, 0x00800000ULL, 0x3f800000ULL, 0x3f800001ULL,
    0x3f800800ULL, 0x7f7fffffULL, 0x7f800000ULL, 0x7f800042ULL
  };
  size_t positive = boundary.size();
  for (size_t i=0;i<positive;++i) {
    boundary.push_back(boundary[i] | (U(1)<<63));
    if (boundary[i] <= 0xffffffffULL) boundary.push_back(boundary[i] | (U(1)<<31));
  }
  auto emit = [&](U a, U b) {
    volatile double x = std::bit_cast<double>(a), y = std::bit_cast<double>(b);
    volatile float p = std::bit_cast<float>(V(a)), q = std::bit_cast<float>(V(b));
    volatile double rounded_product = x*y;
    U expected[] = {
      std::bit_cast<U>(double(x+y)), std::bit_cast<U>(double(x-y)),
      std::bit_cast<U>(double(x*y)), std::bit_cast<U>(double(x/y)),
      U(x==y), U(x<y), std::bit_cast<U>(double(a)),
      std::bit_cast<U>(double(std::bit_cast<int64_t>(a))),
      std::bit_cast<U>(double(p)), U(std::bit_cast<V>(float(x))),
      std::bit_cast<U>(std::floor(x)), std::bit_cast<U>(std::ceil(x)),
      std::bit_cast<U>(std::sqrt(x)),
      std::bit_cast<U>(std::fma(x,y,std::bit_cast<double>((a>>31)|(a<<33)))),
      std::bit_cast<U>(std::fma(x,y,-rounded_product)),
      U(std::bit_cast<V>(float(p+q))), U(std::bit_cast<V>(float(p-q))),
      U(std::bit_cast<V>(float(p*q))), U(std::bit_cast<V>(float(p/q))),
      U(std::bit_cast<V>(std::fma(p,q,std::bit_cast<float>(V((a>>13)|(a<<19)))))),
      U(std::bit_cast<V>(std::fma(p,q,std::bit_cast<float>(V(0x17800000))))),
      U(std::bit_cast<V>(std::fma(p,q,std::bit_cast<float>(V(0x97800000))))),
      std::bit_cast<U>(double((static_cast<unsigned __int128>(b)<<64)|a)),
      std::bit_cast<U>(double(std::bit_cast<__int128>((static_cast<unsigned __int128>(b)<<64)|a)))
    };
    std::fwrite(&a, sizeof(U), 1, out);
    std::fwrite(&b, sizeof(U), 1, out);
    std::fwrite(expected, sizeof(U), 24, out);
  };
  for (U a:boundary) for (U b:boundary) emit(a,b);
  std::mt19937_64 rng(0x4d4f4a4f49454545ULL);
  for (size_t i=0;i<100000;++i) emit(rng(),rng());
  for (size_t i=0;i<25000;++i) {
    U a=rng(); emit(a, (a^(U(1)<<63))+(i%9)-4);
  }
  for (U bit=0;bit<128;++bit) {
    auto base=static_cast<unsigned __int128>(1)<<bit;
    for (int delta=-4;delta<=4;++delta) {
      auto value=base+delta;
      emit(U(value),U(value>>64));
      if(bit>=53) {
        value+=static_cast<unsigned __int128>(1)<<(bit-53);
        emit(U(value),U(value>>64));
      }
    }
  }
  return std::fclose(out) == 0 ? 0 : 2;
}
'''


def main():
    with tempfile.TemporaryDirectory(prefix="mojo-gpu-ieee-") as directory:
        root = Path(directory)
        source = root / "oracle.cpp"
        binary = root / "oracle"
        corpus = root / "expected.bin"
        source.write_text(ORACLE)
        subprocess.run([
            "xcrun", "clang++", "-O2", "-std=c++20", "-ffp-contract=off",
            "-fno-fast-math", str(source), "-o", str(binary)
        ], check=True)
        subprocess.run([str(binary), str(corpus)], check=True)
        subprocess.run([
            "mojo", "run", "-I", ".", "tests/tooling/mojo_gpu_floating.mojo", str(corpus)
        ], check=True)


if __name__ == "__main__":
    main()
