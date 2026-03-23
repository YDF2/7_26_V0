#include <cstdio>
int main()
{
  int count = 0;
  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;
  if (count == 0) return -1;
  cudaDeviceProp prop;
  if (cudaSuccess == cudaGetDeviceProperties(&prop, 0))
    std::printf("%d%d ", prop.major, prop.minor);
  else std::printf("50");
  return 0;
}
