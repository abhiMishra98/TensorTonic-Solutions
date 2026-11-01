# <span style="font-size: 20px;">Fused Softmax</span>

<span style="font-size: 14px;">Numerical stability dictates the structure of any reasonable softmax kernel: the row maximum must be subtracted before any exponential touches a logit, or the exponential overflows fp32 for any input above about $88$ and the entire output collapses to NaN. Triton makes the fusion of max-subtract, exp, sum, and divide into one program almost trivial; the equivalent CUDA kernel needs an explicit block-per-row layout, a shared-memory reduction tree, and at least one $\texttt{\_\_syncthreads}$. The kernel below is the canonical demonstration of why row-level tile programming is the right abstraction for this shape.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given an $(M, N)$ float matrix $x$, the kernel produces an $(M, N)$ output whose rows each sum to $1$:</span>

$$
\texttt{out}[i, j] = \frac{\exp(x[i, j] - \max_k x[i, k])}{\sum_{k=0}^{N-1} \exp(x[i, k] - \max_k x[i, k])}
$$

<span style="font-size: 14px;">The max in the numerator cancels with itself in the denominator, so the result is mathematically identical to the bare $\exp(x) / \sum \exp(x)$. The point of the subtraction is that every exponent argument becomes non-positive, which means $\exp$ never overflows: the largest output is $1$ and the smallest is non-negative.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is one-dimensional with $M$ **programs**, one per row of the input. Each program reads its row index from $\texttt{tl.program\_id(0)}$, offsets into $x$ by $\texttt{row\_idx} \cdot \texttt{x\_row\_stride}$, and operates on the full row as a single $\texttt{BLOCK\_SIZE}$-lane register tile. There is no cross-program communication: rows are independent under softmax, and the per-row computation lives entirely inside one program.</span>

<span style="font-size: 14px;">This is the **row-parallel reduction** pattern, structurally distinct from the per-array reductions earlier in the section. The per-array reduction collapses one input vector into one scalar across many programs; the per-row reduction collapses each row of a matrix into one row of output, with the row reduction happening inside a single program and the cross-row parallelism happening at the grid level. The two shapes look superficially similar but the systems consequences are different: per-row needs no atomics, no scratch, and no cross-program combine, because each program owns a disjoint slice of output. Per-array needs all of those because the output is a single shared scalar.</span>

<span style="font-size: 14px;">The cap on this pattern is that the whole row must fit in one Triton block of registers. For $N$ up to tens of thousands, that is comfortable. For very long rows (hundreds of thousands of columns), the pattern degenerates and the kernel must switch to an online-softmax form that streams the row in chunks while maintaining a running max and running sum. That is the FlashAttention-style kernel and lives in the performance-patterns section of the curriculum.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">The row width $N$ is a runtime value, but Triton tile shapes are compile-time constants. The standard idiom is $\texttt{BLOCK\_SIZE} = \texttt{triton.next\_power\_of\_2}(N)$, computed on the host and passed as a $\texttt{tl.constexpr}$. The compiler uses the constexpr value to size the register tile, unroll the reduction trees ($\texttt{tl.max}$ and $\texttt{tl.sum}$), and pick a vector load width for the row.</span>

<span style="font-size: 14px;">The mask is $\texttt{mask} = \texttt{cols} < N$ on both the $\texttt{tl.load}$ and the $\texttt{tl.store}$, where $\texttt{cols} = \texttt{tl.arange}(0, \texttt{BLOCK\_SIZE})$. The load uses $\texttt{other} = -\texttt{float('inf')}$, not $0$, and the sentinel choice carries the entire correctness argument for non-power-of-two row widths. Masked lanes load $-\infty$, lose every $\texttt{tl.max}$ comparison so the row max is uncorrupted, then under the subtract-and-exp they become $\exp(-\infty - \texttt{row\_max}) = \exp(-\infty) = 0$, contributing nothing to $\texttt{tl.sum}$. The store mask gates the final write so only the real $N$ columns are touched.</span>

<span style="font-size: 14px;">The store side of the mask is as important as the load side. Without it, the kernel would write garbage past column $N$ into the user's output buffer, which is a memory-safety bug rather than a precision bug. Triton's mask discipline puts both sides of the read-write contract under one expression, which is part of why fused kernels at this complexity are still readable.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Each program reads its row from HBM into a register tile in one logical $\texttt{tl.load}$, runs four sequential operations on that tile in registers (max, subtract, exp, sum, divide), and writes the result back to HBM in one logical $\texttt{tl.store}$. The row tile is loaded once and consumed by every subsequent operation; no intermediate value ever spills back to HBM. The whole row-level pipeline reads $N$ fp32 values and writes $N$ fp32 values, which is the bandwidth lower bound for any softmax that produces the full normalized output.</span>

<span style="font-size: 14px;">Contrast with the unfused alternative, which is a three-kernel pipeline: kernel A computes the row max and writes it to a length-$M$ buffer, kernel B reads $x$ and the row max and writes $\exp(x - \texttt{row\_max})$ to a fresh $(M, N)$ intermediate, kernel C reads the intermediate, sums each row, and writes the normalized output. The three-kernel form moves $5 \cdot 4 N M$ bytes of HBM traffic per pass family (two reads of $x$ in pieces, two writes of an $(M, N)$ intermediate, plus the final output), against the fused kernel's $2 \cdot 4 N M$ bytes. The fusion saves roughly $2.5 \times$ the HBM traffic at the same number of operations, on a kernel that is solidly memory-bound. That is the canonical Triton story: the compute is free, the bandwidth is everything, and fusion is how you spend less bandwidth.</span>

<span style="font-size: 14px;">The compiler stages cross-warp partials for both reductions ($\texttt{tl.max}$ for the shift, $\texttt{tl.sum}$ for the normalizer) into shared memory, exactly as it would for a per-array reduction. The author writes nothing about shared memory. Inside one program, the four warps each own a slice of the row tile, reduce their slice via register shuffles, and the compiler combines the warp partials through a short shared-memory hop with an inserted barrier. None of this is visible at the source level.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per output element, the kernel reads $4$ bytes, writes $4$ bytes, and performs roughly three FLOPs (one subtract, one exp, one divide; ignoring the cost of the reductions which amortize over the row). **Arithmetic intensity** is</span>

$$
\frac{3 \text{ FLOPs}}{8 \text{ bytes}} \approx 0.375 \text{ FLOPs/byte}
$$

<span style="font-size: 14px;">which is well under the roofline crossover and places the kernel on the memory-bound side. The exp is technically a transcendental, lowered by the compiler to a small polynomial approximation in fp32, but the cost of that approximation is hidden behind the load latency on a bandwidth-limited kernel. Runtime is set by HBM bandwidth for the two passes over the row data, plus a small launch overhead.</span>

<span style="font-size: 14px;">Numerical stability is the systems concern that drives the whole structure. A naive $\exp(x)$ on a row of logits is correct in infinite precision but useless in fp32: any logit above about $88$ overflows to $+\infty$, the sum becomes $+\infty$, and the divide produces $\texttt{inf} / \texttt{inf} = \texttt{NaN}$ for every column. The max-subtract shifts the largest exponent to $0$, capping every $\exp$ output in $(0, 1]$, and guarantees the row sum is at least $1$ (since the lane with the original max becomes $\exp(0) = 1$). The arithmetic is identical to the bare softmax; the encoding is what makes fp32 hold the answer. This is a generic recipe: any time an exponential reduces a row, the stable form subtracts the max first.</span>

<span style="font-size: 14px;">The test harness has a $\texttt{large\_positive}$ pattern with logits in $[0, 20]$ and a $\texttt{large\_range}$ pattern in $[-50, 50]$, both of which would NaN-out without the max-subtract. They exist specifically to surface the bug class where a kernel passes on small logits and silently fails on production-scale inputs.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author chooses:** the grid of $M$ programs, the constexpr $\texttt{BLOCK\_SIZE} = \texttt{next\_power\_of\_2}(N)$, the row-stride arithmetic, the masks on both load and store, the $-\infty$ sentinel on the load, the four-step in-register pipeline (max, subtract, exp, sum, divide), and the deliberate fusion of all four into one program. The numerical-stability decision (max-subtract) is mathematical, not compiler-discoverable, and lives entirely in the author's hands.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering the row $\texttt{tl.load}$ to coalesced wide HBM transactions, lowering $\texttt{tl.max}$ and $\texttt{tl.sum}$ to warp-shuffle reduction trees with shared-memory staging for the cross-warp combine, lowering $\texttt{tl.exp}$ to a fast fp32 approximation, scheduling the load and the in-register pipeline so that exp evaluation can overlap with the next stage's load latency, and emitting the $\texttt{tl.store}$ as coalesced wide writes. The author writes no shuffle, no shared-memory declaration, and no $\texttt{\_\_syncthreads}$.</span>

<span style="font-size: 14px;">The CUDA contrast is sharper here than in earlier kernels. A hand-written CUDA softmax assigns one block per row, allocates a shared-memory buffer the size of the row (or a tree-reduced version of it), runs an explicit warp-shuffle reduction for the max, syncs, broadcasts the max across the block, syncs again, computes the shifted exp, runs another shared-memory tree reduction for the sum, syncs, broadcasts, and writes the normalized output. The Triton kernel is roughly fifteen lines of Python and reads the entire row into registers in one logical operation. The compiler's job is to compile away the explicit synchronization that the CUDA version writes by hand.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive Triton implementation skips the max-subtract and writes $\exp(x) / \texttt{tl.sum(tl.exp(x))}$. The kernel compiles, passes the $\texttt{ramp}$ test case, and silently fails on the $\texttt{large\_positive}$ pattern with NaN output. The optimization (which is a correctness step, not a performance step) is the max-subtract: one extra $\texttt{tl.max}$ reduction before the exp, plus a lane-wise subtract. The cost is one additional reduction tree inside the program, $\log_2(\texttt{BLOCK\_SIZE})$ register-shuffle steps that finish in well under a microsecond.</span>

<span style="font-size: 14px;">A separate axis of optimization is the choice between the canonical two-reduction kernel above and an **online softmax** that fuses the max and the sum into a single recurrence over the row. The online form maintains a running max $m$ and a running normalizer $\ell$, updating both as the row streams through in chunks: at each new chunk, $m' = \max(m, \max_\text{chunk})$ and $\ell' = \ell \cdot \exp(m - m') + \sum_\text{chunk} \exp(x - m')$. The online form is what FlashAttention generalizes; for a plain softmax on a single matrix, the two-pass (max-then-normalize) form above is simpler and equally fast when the row fits in one tile.</span>

<span style="font-size: 14px;">For very long rows where the row tile would not fit in registers, the kernel must switch to a chunked online form. The current problem's $N$ stays within the single-tile regime, so the simple kernel is the right choice.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take one row $x = [1, 2, 3]$ with $N = 3$. The host computes $\texttt{BLOCK\_SIZE} = \texttt{next\_power\_of\_2}(3) = 4$ and launches one program for that row.</span>

<span style="font-size: 14px;">The program builds $\texttt{cols} = [0, 1, 2, 3]$ and $\texttt{mask} = [\texttt{T}, \texttt{T}, \texttt{T}, \texttt{F}]$. The $\texttt{tl.load}$ with $\texttt{other} = -\infty$ produces $\texttt{row} = [1, 2, 3, -\infty]$. The first reduction is $\texttt{row\_max} = \texttt{tl.max(row)} = 3$, since $-\infty$ loses every comparison. The lane-wise subtract gives $\texttt{row} = [-2, -1, 0, -\infty]$. The lane-wise exp produces $[e^{-2}, e^{-1}, 1, 0]$, where the masked lane becomes exactly $0$ because $\exp(-\infty) = 0$. The second reduction is $\texttt{row\_sum} = e^{-2} + e^{-1} + 1 \approx 1.503$. The lane-wise divide gives roughly $[0.090, 0.245, 0.665, 0]$. The $\texttt{tl.store}$ with the same mask writes only the first three lanes back to $\texttt{out}$.</span>

<span style="font-size: 14px;">Now contrast with the broken sentinel: if the load had used $\texttt{other=0.0}$, the row would have been $[1, 2, 3, 0]$ and $\texttt{tl.max}$ would still give $3$, so the row max happens to survive. The bug appears one step later: after subtract, the masked lane is $0 - 3 = -3$, and $\exp(-3) \approx 0.05$ contributes spuriously to the sum, shrinking every output column by a factor of about $1.03$. On an all-negative row, the bug is even worse because the masked-zero lane becomes the new row max. This is exactly the kind of silent precision bug that $-\infty$ sentinels exist to prevent.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Skipping max-subtract.** A bare $\exp(x)$ overflows fp32 above about $88$. The output is $\texttt{inf} / \texttt{inf} = \texttt{NaN}$ for every column whose row contains a large positive logit. The fix is one $\texttt{tl.max}$ and one subtract, and it is a correctness fix, not a performance one.</span>
* <span style="font-size: 14px;">**Using $\texttt{other=0.0}$ on the row mask.** Masked lanes feed $0$ into $\texttt{tl.max}$, which corrupts the row max for any all-negative row. After subtract-and-exp the masked lane contributes $\exp(0 - \texttt{row\_max})$ to the sum, also wrong. The sentinel must be $-\infty$ so the masked lane loses the max comparison and exponentiates to $0$.</span>
* <span style="font-size: 14px;">**Hardcoding the row stride.** Writing $\texttt{row\_idx} \cdot N$ instead of $\texttt{row\_idx} \cdot \texttt{x\_row\_stride}$ works only when $x$ is row-contiguous with no padding. Passing the stride from the host keeps the kernel general against transposed or sliced inputs.</span>
* <span style="font-size: 14px;">**Forgetting the store mask.** A missing $\texttt{mask}$ on $\texttt{tl.store}$ writes garbage past column $N$ into the user's output buffer. The load mask is more often remembered (it shows up first); the store mask is the one that silently corrupts neighboring memory if dropped.</span>

---