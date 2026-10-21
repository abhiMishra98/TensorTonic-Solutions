import torch
import triton
import triton.language as tl


@triton.jit
def matmul_qk_kernel(
    Q_ptr, K_ptr, S_ptr,
    Nq,Nk, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # Write code here
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    
    offs_m = start_m + tl.arange(0, BLOCK_M) #row indices from the entire block
    offs_n = start_n + tl.arange(0, BLOCK_N) #column indices from the entire block
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q_ptr + offs_m[:,None] * D + offs_d[None,:]
    #TO-DO : Add mask
    q_mask = (offs_m[:, None] < Nq) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs,mask= q_mask,other=0.0)

    k_ptrs = K_ptr + offs_n[None,:]*D + offs_d[:,None]
    k_mask =  (offs_n[None, :] < Nk) & (offs_d[:, None] < D)
    k = tl.load(k_ptrs,mask= k_mask,other=0.0) 

    S = tl.dot(q,k)
    S = S * scale
    s_ptrs = S_ptr + offs_m[:,None]*Nk + offs_n[None,:]
    s_mask = (offs_m[:,None] < Nq) & (offs_n[None,:] < Nk)
    tl.store(s_ptrs,S,mask=s_mask)
    pass


@triton.jit
def softmax_rows_kernel(
    S_ptr, N, BLOCK_N: tl.constexpr,
):
    # Write code here
    pid = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)

    s_ptrs = S_ptr + pid*N + offs_n
    mask = offs_n < N
    s = tl.load(s_ptrs, mask = mask, other=0.0)
    s = tl.where(mask,s,float("-inf"))
    max = tl.max(s)
    exp = tl.exp(s - max)
    s = tl.where(mask,exp,0)
    den = tl.sum(exp)
    s = exp / den
    tl.store(s_ptrs,s,mask=mask)
    
    pass


@triton.jit
def matmul_av_kernel(
    S_ptr, V_ptr, out_ptr,
    Nq,Nk, D,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Write code here
    row = tl.program_id(0)
    offs_m = row * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for start_n in range(0, Nk, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        s_ptrs = S_ptr + offs_m[:, None] * Nk + offs_n[None, :]
        v_ptrs = V_ptr + offs_n[:, None] * D + offs_d[None, :]
        s_mask = (offs_m[:, None] < Nq) & (offs_n[None, :] < Nk)
        v_mask = (offs_n[:, None] < Nk) & (offs_d[None, :] < D)
        s = tl.load(s_ptrs, mask=s_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        o += tl.dot(s, v, input_precision="ieee")

    o_ptrs = out_ptr + offs_m[:, None] * D + offs_d[None, :]
    o_mask = (offs_m[:, None] < Nq) & (offs_d[None, :] < D)
    tl.store(o_ptrs, o, mask=o_mask)
    pass


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the three-stage attention: QK^T -> softmax -> @V."""
    Nq, D = Q.shape
    Nk = K.shape[0]
    scale = 1.0 / (D ** 0.5)
    S = torch.empty((Nq, Nk), device=Q.device, dtype=torch.float32)

    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_D = 16
    while BLOCK_D < D:
        BLOCK_D *= 2
    BLOCK_N_PADDED = 1
    while BLOCK_N_PADDED < Nk: #Softmax is (Nq,Nk) now, softmax happens across K col entries
        BLOCK_N_PADDED *= 2

    grid_qk = (triton.cdiv(Nq,BLOCK_M),triton.cdiv(Nk,BLOCK_N))
    matmul_qk_kernel[grid_qk](
        Q, K, S, Nq, Nk, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    softmax_rows_kernel[(Nq,)](S, Nk, BLOCK_N=BLOCK_N_PADDED)

    grid_av = (triton.cdiv(Nq,BLOCK_M),)
    matmul_av_kernel[grid_av](
        S, V, out, Nq,Nk, D,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N,
    )