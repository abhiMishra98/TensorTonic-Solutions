# <span style="font-size: 20px;">Scaled Dot-Product Attention</span>

<span style="font-size: 14px;">Three kernels, one pipeline, one $N \times N$ scratch buffer. This is the pedagogical baseline for attention: the literal translation of the textbook formula into Triton, with each of the three stages running as its own launch and the score matrix materialized in HBM between them. The parallel pattern is a **2D tile matmul, followed by a per-row reduction, followed by a 2D tile matmul**. Same final answer as FlashAttention, very different memory footprint. Studying this kernel first makes the fused alternative's payoff concrete.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For $Q, K, V \in \mathbb{R}^{N \times D}$, the output $\texttt{out} \in \mathbb{R}^{N \times D}$ is</span>

$$
\texttt{out} = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{D}}\right) V
$$

<span style="font-size: 14px;">The three stages are factored explicitly: stage 1 produces $S = Q K^\top / \sqrt{D} \in \mathbb{R}^{N \times N}$, stage 2 applies row-wise softmax to $S$ in place, stage 3 produces $\texttt{out} = S \cdot V$. Each stage is a separate $\texttt{@triton.jit}$ kernel with a separate launch grid.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">Stage 1, the $QK^\top$ matmul, runs on a 2D grid of $\lceil N / \texttt{BLOCK\_M} \rceil \times \lceil N / \texttt{BLOCK\_N} \rceil$ **programs**. Each program owns one `(BLOCK_M, BLOCK_N)` output tile of the score matrix and reduces the full head-dimension $D$ inside the program by loading aligned tiles of $Q$ and $K$. The grid is over the output, not over the reduction; the reduction is internal to each program.</span>

<span style="font-size: 14px;">Stage 2, the row-wise softmax, runs on a 1D grid of $N$ programs, one per row of the score matrix. Each program loads the whole row in a single tile of size $\texttt{BLOCK\_N\_PADDED} = \texttt{next\_power\_of\_2}(N)$, applies the max-subtract softmax in registers, and writes the row back in place. This is the textbook fused softmax pattern.</span>

<span style="font-size: 14px;">Stage 3, the $\mathrm{attn} \cdot V$ matmul, runs on a 1D grid of $\lceil N / \texttt{BLOCK\_M} \rceil$ programs. Each program owns one `BLOCK_M`-row block of the output, accumulates a `(BLOCK_M, D)` register tile across an inner loop over the $N$ axis in $\texttt{BLOCK\_N}$ chunks, and stores the result. The output dimension $D$ is the inner head dimension and is held whole in registers per program via $\texttt{BLOCK\_D} = \texttt{next\_power\_of\_2}(D)$.</span>

<span style="font-size: 14px;">Across the three stages, three distinct parallel patterns appear: a 2D output-tiled matmul (stage 1), a 1D per-row reduction (stage 2), and a 1D row-block matmul with an inner $N$ reduction (stage 3). Each is a clean pedagogical example in its own right; combining them into one launch is what FlashAttention does. Studying the three separately first makes the fused alternative legible: the score tile that stage 1 writes and stage 3 reads is exactly the tile that FlashAttention holds in registers instead.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">Block sizes are constexpr meta-parameters. Stage 1 picks $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 16$ for the score output tile and $\texttt{BLOCK\_D}$ rounded up to the next power of two for the head dimension; the latter is held whole in registers so the inner reduction is a single sum over the head axis. Stage 2 must use $\texttt{BLOCK\_N\_PADDED} \ge N$ so the entire row fits in one tile inside the softmax kernel; the choice is $\texttt{next\_power\_of\_2}(N)$. Stage 3 mirrors stage 1 with $\texttt{BLOCK\_M}, \texttt{BLOCK\_N}, \texttt{BLOCK\_D}$, but the reduction is over $N$ rather than $D$.</span>

<span style="font-size: 14px;">Mask discipline is non-trivial because three independent kernels each have a tail. Stage 1 needs row, column, and head masks on the $Q$ and $K$ loads and on the $S$ store. Stage 2 needs a $\texttt{cols} < N$ mask on the row load and store, plus the canonical $\texttt{other} = -10^{30}$ on the load so padded lanes do not pull the row max down. Stage 3 needs row, head, and inner column masks. The subtlest is stage 2: if the load uses $\texttt{other} = 0$, padded lanes contribute weight 1 after exp and shift the entire softmax row. A correctness bug, not a performance bug.</span>

<span style="font-size: 14px;">Stage 3's inner column mask deserves a separate note. The kernel walks the $N$ axis in $\texttt{BLOCK\_N}$ chunks; the last chunk may overshoot $N$. The mask zeros out the masked lanes on the score-tile load (so they contribute nothing to the matrix-vector accumulation), and the same mask gates the $V$ load. There is no need to fill the masked $V$ lanes with $-\infty$ here because the score has already been normalized by stage 2 to a clean weight distribution; the only requirement is that masked lanes contribute zero to the output accumulator, which $\texttt{other} = 0$ achieves directly.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">The scratch tensor is the defining cost of this kernel. The launcher calls $\texttt{torch.empty}((N, N))$ before stage 1 and lets PyTorch's caching allocator manage it; when $\texttt{solve}$ returns, the tensor is dropped and its memory goes back to the pool. For $N = 4096$ in fp32 that scratch is $4096^2 \cdot 4 = 64$ MB. Stage 1 writes it, stage 2 reads and rewrites it, stage 3 reads it again. Total $N^2$ traffic across the three stages is roughly $4 \cdot 4 N^2$ bytes, which for $N = 4096$ is $256$ MB of pure score traffic.</span>

<span style="font-size: 14px;">Within each program, the standard tiled-matmul reuse story applies. Stage 1's program loads its `(BLOCK_M, BLOCK_D)` tile of $Q$ once and its `(BLOCK_N, BLOCK_D)` tile of $K$ once, then performs a broadcast-multiply-sum across the head axis to fill its `(BLOCK_M, BLOCK_N)` score tile. The $Q$ tile is reused $\texttt{BLOCK\_N}$ times and the $K$ tile $\texttt{BLOCK\_M}$ times, raising the in-program arithmetic intensity to $\Theta(\texttt{BLOCK\_D})$ FLOPs per byte. Stage 3 has the same shape with the roles permuted.</span>

<span style="font-size: 14px;">The bottleneck is not in-program reuse, it is between-stage traffic. The score matrix is the dominant operand and it lives in HBM for the lifetime of the pipeline. L2 helps a little: if the scratch fits in L2 (typical only for $N$ in the hundreds), the inter-stage reads may hit cache. For inference-scale $N$ in the thousands, the scratch overflows L2 and every score round-trip pays full HBM latency.</span>

<span style="font-size: 14px;">A useful accounting: at $N = 1024, D = 64$ the inputs $Q, K, V$ together are $3 \cdot 1024 \cdot 64 \cdot 4 = 768$ KB, the output is $256$ KB, and the score scratch is $4$ MB. The scratch is five times larger than all the inputs combined, and the three stages move it through HBM four times total ($1 \times$ write in stage 1, $1 \times$ read and $1 \times$ write in stage 2, $1 \times$ read in stage 3). The total HBM traffic is dominated by the scratch by a factor of more than $10\times$ at this shape, and the imbalance widens with $N$.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">The pipeline is **bandwidth-dominated** at moderate to large $N$. Per output element, the kernel reads $O(D)$ values to compute the matmul rows and reads or writes $O(N)$ values through the score matrix; the score traffic scales as $N^2$ while the FLOPs scale as $N^2 \cdot D$, so the arithmetic intensity is $\Theta(D)$ FLOPs per byte of score traffic. For $D = 64$ that puts the pipeline near or below the roofline crossover on most accelerators, with the score round-trips bounding throughput.</span>

<span style="font-size: 14px;">The contrast with FlashAttention is exactly this: by collapsing the three stages into one program that holds the score tile in registers, the $N^2$ HBM traffic vanishes and the kernel moves cleanly to the compute-bound side of the roofline. The three-stage formulation cannot recover that bandwidth no matter how much each individual stage is tuned, because the architectural cost is the inter-stage materialization itself.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author handles:** the three-kernel decomposition, the scratch allocation, the constexpr block sizes per stage, the mask predicates including the $-10^{30}$ fill on the softmax load, the max-shift in stage 2, the head-dimension reduction inside the matmul programs, and the in-place semantic for the softmax that lets the kernel write back to the same buffer it just read from. The choice to factor as three kernels is itself the author's call; an equally correct formulation would fuse stages into different boundaries.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering each tile load and store to wide vector PTX, allocating registers for the score and head-dim tiles, choosing the inner schedule for the broadcast-multiply-sum that implements the matmul (the canonical reference uses elementwise multiply with $\texttt{tl.sum}$ rather than $\texttt{tl.dot}$, again because the small tile sizes do not engage tensor cores cleanly), and inserting any pipelining between successive loads inside a kernel. The compiler also chooses how the per-program tiles are sharded across warps internally; the author never names a warp count unless they autotune over it.</span>

<span style="font-size: 14px;">One interface decision is worth highlighting: the scratch allocation lives in the Python launcher, not inside any kernel. The kernels see the scratch as a regular tensor pointer with a $N \cdot N$ stride layout. This is the standard Triton pattern: PyTorch's caching allocator manages the buffer's lifetime, and the launcher passes it across stages by reference. There is no Triton-side memory management because there does not need to be one; the framework handles allocation and the compiler handles the pointer arithmetic.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">Within the three-stage formulation, the canonical reference is already a reasonable baseline. Optimizations that stay inside the formulation: switch stage 1 and stage 3 to $\texttt{tl.dot}$ with larger block sizes ($\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 64$ or $128$) to engage tensor cores, autotune over the block sizes for the active head-dimension and sequence length, and pipeline the inner head-axis loop in stage 1. Each of these can buy single-digit to low-double-digit percentage improvements on modern hardware.</span>

<span style="font-size: 14px;">None of these optimizations remove the $N^2$ HBM round-trip. The only way to remove it is to fuse the three stages into one kernel that holds the score tile in registers across the softmax and the second matmul, which is FlashAttention. The three-stage kernel is therefore best viewed as the maximum-readability baseline and the upper bound on HBM traffic: anything more sophisticated than this is moving toward fusion.</span>

<span style="font-size: 14px;">A partial fusion is also possible and instructive: combine stages 2 and 3 into one kernel that streams the score row through registers as it computes the output row. This eliminates one HBM round-trip on the scratch but still requires stage 1 to materialize the full $N \times N$ matrix, so the dominant bandwidth cost remains. Full fusion of all three stages into FlashAttention requires the online softmax recurrence, because there is no way to read $K, V$ once each and still compute the exact softmax without either two passes over the score row or a one-pass running-max formulation. The recurrence is the missing piece, and that is the reason FlashAttention took until 2022 to crystallize even though tiled matmul and fused softmax were standard techniques years earlier.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 4, D = 2$, $\texttt{BLOCK\_M} = \texttt{BLOCK\_N} = 2$, $\texttt{BLOCK\_D} = 2$. Stage 1 launches a $2 \times 2$ grid of four programs, one per `(2, 2)` score tile. Program $(0, 0)$ loads the first two rows of $Q$ and the first two rows of $K$, broadcasts and multiplies to form a `(2, 2, 2)` intermediate, sums along the head axis, scales by $1/\sqrt{2}$, and stores the result into $S[0:2, 0:2]$. The other three programs do the same for their tiles. Total writes to the scratch: $16$ fp32 elements, $64$ bytes.</span>

<span style="font-size: 14px;">Stage 2 launches four programs, one per row. Program $0$ reads $S[0, 0:4]$ into a tile of size $\texttt{BLOCK\_N\_PADDED} = 4$ with $\texttt{other} = -10^{30}$, takes the max, subtracts, exponentiates, sums, divides, and writes the normalized row back to $S[0, 0:4]$. Total scratch traffic in stage 2: $16$ elements read and $16$ written, $128$ bytes.</span>

<span style="font-size: 14px;">Stage 3 launches two programs (since $\lceil 4 / \texttt{BLOCK\_M} \rceil = 2$). Program $0$ loads $S[0:2, :]$ in two $2$-wide column chunks, loads the matching rows of $V$, accumulates a `(2, 2)` output tile in registers, and stores it to $\texttt{out}[0:2, :]$. Total scratch reads in stage 3: $16$ elements, $64$ bytes. Aggregate score traffic across the three stages: $64 + 128 + 64 = 256$ bytes for a $4 \times 4$ matrix. The same accounting at $N = 4096$ gives $256$ MB.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Padded softmax lanes loaded as zero.** Stage 2's $\texttt{tl.load}$ must use $\texttt{other} = -10^{30}$, not the default zero. Zero entries become weight $1$ after the exp and skew every row whose length is not a multiple of the padded block size.</span>
* <span style="font-size: 14px;">**$\texttt{BLOCK\_N\_PADDED}$ smaller than $N$ in stage 2.** The softmax kernel loads the whole row in a single $\texttt{tl.arange}(0, \texttt{BLOCK\_N\_PADDED})$. If $\texttt{BLOCK\_N\_PADDED} < N$ the tail of the row is invisible to the max and the sum, and the row is normalized over a prefix only.</span>
* <span style="font-size: 14px;">**Scaling by $1/\sqrt{D}$ after the softmax.** Softmax is not linear, so scaling the output rows is mathematically wrong. The divide must happen on the score tile before the exp, which is why stage 1 takes $\texttt{scale}$ as an argument.</span>
* <span style="font-size: 14px;">**Treating the scratch as persistent across calls.** The $(N, N)$ buffer is allocated per call. Caching it globally breaks the test harness, which invokes $\texttt{solve}$ with varying $N$; the global would either be too small for a larger row or hold stale data from a previous call.</span>

---