import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, gamma_ptr, out_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Write code here
    pId = tl.program_id(0)
    start_n = pId*stride_x_row
    offs_n = tl.arange(0,BLOCK_SIZE)
    x_ptrs = x_ptr + start_n + offs_n
    mask = offs_n < N
    x = tl.load(x_ptrs,mask=mask, other = 0)

    gamma_ptrs = gamma_ptr + tl.arange(0,BLOCK_SIZE)
    gamma = tl.load(gamma_ptrs, mask=mask, other=0)
    
    sq = x * x
    ms = tl.sum(sq,axis=0,keep_dims=True) / N
    r = tl.sqrt(ms + eps)
    xhat = (x / r)*gamma
    o_ptrs = out_ptr + pId*stride_out_row + tl.arange(0,BLOCK_SIZE)
    tl.store(o_ptrs,xhat,mask=mask)
    pass


def solve(x: torch.Tensor, gamma: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    """Launch the RMSNorm forward kernel: one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    rmsnorm_fwd_kernel[grid](
        x, gamma, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )