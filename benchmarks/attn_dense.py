import sys, time, torch
sys.path.insert(0, "/var/tmp/sglang-sxm2/python")
from sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda import sm70_cuda_decode_partial
dev="cuda:0"; B,H,D,HKV,PS=1,24,256,4,16; SCALE=D**-0.5; N=262144
gb=2*N*HKV*D
def run(k,v,pt,label):
    q=torch.randn(B,H,D,dtype=torch.float16,device=dev)*SCALE
    sl=torch.tensor([N],dtype=torch.int32,device=dev)
    splits=320
    for _ in range(3): sm70_cuda_decode_partial(q,k,v,pt,sl,splits,32,SCALE,1.0,1.0)
    torch.cuda.synchronize(); t0=time.perf_counter(); reps=20
    for _ in range(reps): sm70_cuda_decode_partial(q,k,v,pt,sl,splits,32,SCALE,1.0,1.0)
    torch.cuda.synchronize(); ms=(time.perf_counter()-t0)/reps*1e3
    print(f"{label:14s} {ms:7.3f} ms {gb/1e9/(ms/1e3):7.1f} GB/s")
npages=N//PS+1
kp=torch.randn(npages,PS,HKV,D,dtype=torch.float16,device=dev).to(torch.float8_e5m2)
vp=torch.randn(npages,PS,HKV,D,dtype=torch.float16,device=dev).to(torch.float8_e5m2)
pt=torch.arange(N//PS,dtype=torch.int32,device=dev).unsqueeze(0)
run(kp,vp,pt,"paged")
kd=torch.randn(1,N,HKV,D,dtype=torch.float16,device=dev).to(torch.float8_e5m2)
vd=torch.randn(1,N,HKV,D,dtype=torch.float16,device=dev).to(torch.float8_e5m2)
empty=torch.empty(0,dtype=torch.int32,device=dev)
run(kd,vd,empty,"dense (no PT)")
