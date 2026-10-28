import sys, torch
sys.path.insert(0, "/var/tmp/sglang-sxm2/python")
from sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda import sm70_cuda_decode_partial
from sglang.srt.layers.attention.tilelang_fa_v100._kernels_paged_decode import _decode_combine_kernel
dev="cuda:0"; B,H,D,HKV,PS=1,24,256,4,16; SCALE=D**-0.5; N=2048
splits=40; tps=32
torch.manual_seed(0)
k=torch.randn(1,N,HKV,D,dtype=torch.float16,device=dev)
v=torch.randn(1,N,HKV,D,dtype=torch.float16,device=dev)
q=torch.randn(B,H,D,dtype=torch.float16,device=dev)
kc=k.to(torch.float8_e5m2); vc=v.to(torch.float8_e5m2)
empty=torch.empty(0,dtype=torch.int32,device=dev)
sl=torch.tensor([N],dtype=torch.int32,device=dev)
po,pl=sm70_cuda_decode_partial(q,kc,vc,empty,sl,splits,tps,SCALE,1.0,1.0)
combine=_decode_combine_kernel(B,H,D,splits,256,tps)
out=combine(po,pl,sl)
# reference in fp32 from the same quantized KV
kf=kc.to(torch.float32); vf=vc.to(torch.float32)
ref=torch.empty(B,H,D,dtype=torch.float32,device=dev)
G=H//HKV
for h in range(HKV):
    kk=kf[0,:,h,:]; vv=vf[0,:,h,:]
    for g in range(G):
        qq=q[0,h*G+g].float()*SCALE
        logits=kk@qq
        p=torch.softmax(logits,dim=0)
        ref[0,h*G+g]=p@vv
d=(out.float()-ref).abs()
print(f"max|err|={d.max().item():.5f} mean|err|={d.mean().item():.6f} ref_rms={ref.pow(2).mean().sqrt().item():.4f}")
rel=(d/ref.abs().clamp_min(1e-3)).max().item()
print(f"max rel={rel:.4f}")
