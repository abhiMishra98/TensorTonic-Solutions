import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here

    pid = tl.program_id(0)
    startAddr = pid * BLOCK_SIZE
    ySum = 0.0   
    offsetAddr = startAddr + tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + offsetAddr
    mask = offsetAddr < n
    x = tl.load(x_ptrs,mask = mask, other = 0.0)
    rSum = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr, rSum)
    
    pass


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch sum_kernel on the provided tensors."""
    n = x.numel()
    out.zero_()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    sum_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)