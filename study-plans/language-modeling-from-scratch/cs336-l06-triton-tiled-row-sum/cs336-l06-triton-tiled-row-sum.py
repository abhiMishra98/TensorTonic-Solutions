import torch
import triton
import triton.language as tl

@triton.jit
def tiled_row_sum_kernel(x_ptr, out_ptr, n_cols, row_stride, col_stride, BLOCK_SIZE: tl.constexpr):
    """
    Returns nothing; writes one float32 row sum through out_ptr.
    """
    pId = tl.program_id(0)
    start = pId * row_stride
    y = 0.0
    for i in range(0,n_cols,BLOCK_SIZE):
        offset = (i + tl.arange(0, BLOCK_SIZE) )* col_stride
        x_ptrs = x_ptr + start + offset
        mask = i + tl.arange(0, BLOCK_SIZE) < n_cols
        x = tl.load(x_ptrs, mask=mask, other=0)
        x = x.to(tl.float32)
        y += tl.sum(x, axis = 0)
   
    out_ptrs = out_ptr + pId
    tl.store(out_ptrs,y)
    pass

def solve(x: torch.Tensor, out: torch.Tensor, tile_size: int) -> None:
    if x.ndim != 2 or out.ndim != 1 or out.shape[0] != x.shape[0]:
        raise ValueError("x must be a matrix and out must have one value per row")
    if not x.is_cuda or not out.is_cuda or x.device != out.device:
        raise ValueError("x and out must be CUDA tensors on the same device")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16) or out.dtype != torch.float32:
        raise ValueError("x has an unsupported dtype or out is not float32")
    if tile_size < 32 or tile_size > 1024 or tile_size & (tile_size - 1):
        raise ValueError("tile_size must be a power of two from 32 through 1024")
    if x.shape[0] == 0:
        return
    if x.shape[1] == 0:
        out.zero_()
        return
    kernel_x = x.to(torch.float32) if x.dtype == torch.bfloat16 else x
    grid = (x.shape[0],)
    tiled_row_sum_kernel[grid](
        kernel_x,
        out,
        x.shape[1],
        kernel_x.stride(0),
        kernel_x.stride(1),
        BLOCK_SIZE=tile_size,
    )
