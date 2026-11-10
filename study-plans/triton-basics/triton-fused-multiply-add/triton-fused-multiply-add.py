import torch
import triton
import triton.language as tl


@triton.jit
def fma_kernel(x_ptr, y_ptr, out_ptr, n, a, BLOCK_SIZE: tl.constexpr):
    # Write code here

    pid = tl.program_id(0)
    startAddr = pid * BLOCK_SIZE
    offs = startAddr + tl.arange(0, BLOCK_SIZE)
   
    x_ptrs = x_ptr + offs
    mask = offs < n
    x = tl.load(x_ptrs, mask=mask, other = 0.0)

    y_ptrs = y_ptr + offs
    y = tl.load(y_ptrs, mask = mask, other = 0.0)
    yOut = tl.fma(a,x,y)

    out_ptrs = out_ptr + offs
    tl.store(out_ptrs, yOut, mask = mask)
    pass


def solve(a: float, x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Launch fma_kernel: out = a * x + y."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fma_kernel[grid](x, y, out, n, a, BLOCK_SIZE=BLOCK_SIZE)