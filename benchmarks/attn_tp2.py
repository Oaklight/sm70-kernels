"""In-service shape: TP2 means 2 kv heads and 12 q heads per CTA."""
import sys, time, torch
sys.path.insert(0, "/var/tmp/sglang-sxm2/python")
from sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda import sm70_cuda_decode_partial
dev="cuda:0"; B,H,D,HKV,PS=1,12,256,2,16; SCALE=D**-0.5
print(f"TP2 shape: heads={H} kv_heads={HKV}")
for N in (131072, 262144):
    splits = max(1, -(-160 // (B*HKV)))
    npages=N//PS+1
    k=torch.randn(npages,PS,HKV,D,dtype=torch.float16,device=dev).to(torch.float8_e5m2)
    v=torch.randn(npages,PS,HKV,D,dtype=torch.float16,device=dev).to(torch.float8_e5m2)
    q=torch.randn(B,H,D,dtype=torch.float16,device=dev)*SCALE
    pt=torch.arange(N//PS,dtype=torch.int32,device=dev).unsqueeze(0)
    sl=torch.tensor([N],dtype=torch.int32,device=dev)
    gb=2*N*HKV*D
    for _ in range(3): sm70_cuda_decode_partial(q,k,v,pt,sl,splits,32,SCALE,1.0,1.0)
    torch.cuda.synchronize(); t0=time.perf_counter(); reps=20
    for _ in range(reps): sm70_cuda_decode_partial(q,k,v,pt,sl,splits,32,SCALE,1.0,1.0)
    torch.cuda.synchronize(); ms=(time.perf_counter()-t0)/reps*1e3
    print(f"N={N:>7d} splits={splits:>3d} {ms:7.3f} ms  {gb/1e9/(ms/1e3):7.1f} GB/s  16 layers => {ms*16:6.2f} ms/token")
