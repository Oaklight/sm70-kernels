import sys, time, torch
sys.path.insert(0, "/var/tmp/sglang-sxm2/python")
from sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda import sm70_cuda_decode_partial
dev="cuda:0"; B,H,D,HKV,PS=1,24,256,4,16; SCALE=D**-0.5
print(f"device={torch.cuda.get_device_name(0)}")
for N in (262144,):
    npages=N//PS+1
    k=torch.randn(npages,PS,HKV,D,dtype=torch.float16,device=dev).to(torch.float8_e5m2)
    v=torch.randn(npages,PS,HKV,D,dtype=torch.float16,device=dev).to(torch.float8_e5m2)
    q=torch.randn(B,H,D,dtype=torch.float16,device=dev)*SCALE
    pt=torch.arange(N//PS,dtype=torch.int32,device=dev).unsqueeze(0)
    sl=torch.tensor([N],dtype=torch.int32,device=dev)
    gb=2*N*HKV*D
    for splits in (10,20,40,80,160,320,640,1280):
        for _ in range(3): sm70_cuda_decode_partial(q,k,v,pt,sl,splits,32,SCALE,1.0,1.0)
        torch.cuda.synchronize(); t0=time.perf_counter(); reps=20
        for _ in range(reps): sm70_cuda_decode_partial(q,k,v,pt,sl,splits,32,SCALE,1.0,1.0)
        torch.cuda.synchronize(); ms=(time.perf_counter()-t0)/reps*1e3
        print(f"splits={splits:>5d} {ms:7.3f} ms {gb/1e9/(ms/1e3):7.1f} GB/s")
