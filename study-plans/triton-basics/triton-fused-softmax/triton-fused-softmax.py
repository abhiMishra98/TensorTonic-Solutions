import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pId = tl.program_id(0)
    start_n = x_row_stride * pId
    offs_n = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + start_n + offs_n
    mask = offs_n < n_cols
    x = tl.load(x_ptrs, mask = mask, other=float('-inf'))
    max = tl.max(x,keep_dims = True)
    diff = x - max
    num = tl.exp(diff)
    den = tl.sum(num, axis =0, keep_dims = True)

    out = num/den
    out_ptrs = out_ptr + out_row_stride*pId + tl.arange(0, BLOCK_SIZE)
    mask = offs_n < n_cols
    tl.store(out_ptrs,out, mask = mask)
    pass


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )