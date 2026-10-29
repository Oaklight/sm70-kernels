// HMMA against HFMA2 on this SM70 part, in MACs per cycle per SM.
//
// The decode kernel's compute phase is instruction-issue-bound: QK+PV need 1,536
// warp-instructions of HFMA2 per tile per CTA against an issue floor of 768
// cycles, and the phase measures 2,132 cycles on the SM. If one MMA instruction
// does materially more multiply-accumulates than one HFMA2, that is the only
// lever that changes the count rather than rearranging it.
//
// Both loops are dependency chains on their own accumulators, so neither is
// measuring an empty pipeline.
//
// Build and run (SM70 only):
//   nvcc -O3 -arch=sm_70 -o mma_probe mma_probe.cu && ./mma_probe
//
// Measured on lambda12, Tesla V100-SXM2-32GB:
//   hfma2 4.472 ms  14655.4 GMAC/s  119.7 MAC/cycle/SM
//   hmma 18.546 ms  56540.2 GMAC/s  461.9 MAC/cycle/SM
//   ratio hmma/hfma2 = 3.86x
#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

__global__ void k_hfma2(__half2* out, int iters) {
  const __half2 a = __float2half2_rn(1.0f);
  const __half2 b = __float2half2_rn(1.0001f);
  __half2 c0 = a, c1 = a, c2 = a, c3 = a;
  for (int i = 0; i < iters; ++i) {
    c0 = __hfma2(a, b, c0);
    c1 = __hfma2(a, b, c1);
    c2 = __hfma2(a, b, c2);
    c3 = __hfma2(a, b, c3);
  }
  out[threadIdx.x] = __hadd2(__hadd2(c0, c1), __hadd2(c2, c3));
}

__global__ void k_hmma(float* out, int iters) {
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
  wmma::fill_fragment(acc, 0.0f);
  wmma::fill_fragment(fa, __float2half(1.0f));
  wmma::fill_fragment(fb, __float2half(1.0f));
  for (int i = 0; i < iters; ++i) {
    wmma::mma_sync(acc, fa, fb, acc);
  }
  out[threadIdx.x] = acc.x[0];
}

int main() {
  const int blocks = 80, threads = 512, iters = 200000;
  __half2* ho;
  float* fo;
  cudaMalloc(&ho, threads * sizeof(__half2));
  cudaMalloc(&fo, threads * sizeof(float));

  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);

  // Warm the clock: a V100 idles near 1300 MHz and only boosts under load.
  for (int i = 0; i < 50; ++i) {
    k_hfma2<<<blocks, threads>>>(ho, iters / 20);
    k_hmma<<<blocks, threads>>>(fo, iters / 20);
  }
  cudaDeviceSynchronize();

  float ms_h = 0.f, ms_m = 0.f;
  for (int r = 0; r < 5; ++r) {
    cudaEventRecord(a);
    k_hfma2<<<blocks, threads>>>(ho, iters);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float t;
    cudaEventElapsedTime(&t, a, b);
    if (ms_h == 0.f || t < ms_h) ms_h = t;

    cudaEventRecord(a);
    k_hmma<<<blocks, threads>>>(fo, iters);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    cudaEventElapsedTime(&t, a, b);
    if (ms_m == 0.f || t < ms_m) ms_m = t;
  }

  // HFMA2: 4 instructions per iteration per thread, 2 MAC each.
  // HMMA:  1 wmma per iteration per warp, 16x16x16 = 4096 MAC across 32 lanes.
  const double macs_h = (double)blocks * threads * iters * 4 * 2;
  const double macs_m = (double)blocks * (threads / 32) * iters * 4096;
  const double ghz = 1.53;  // boost clock; the ratio does not depend on it
  const double sms = 80;
  const double cyc_h = ms_h * 1e-3 * ghz * 1e9;
  const double cyc_m = ms_m * 1e-3 * ghz * 1e9;

  printf("hfma2 %.3f ms  %.1f GMAC/s  %.1f MAC/cycle/SM\n", ms_h,
         macs_h / ms_h / 1e6, macs_h / cyc_h / sms);
  printf("hmma  %.3f ms  %.1f GMAC/s  %.1f MAC/cycle/SM\n", ms_m,
         macs_m / ms_m / 1e6, macs_m / cyc_m / sms);
  printf("ratio hmma/hfma2 = %.2fx\n", macs_m / ms_m / (macs_h / ms_h));
  return 0;
}
