# <span style="font-size: 20px;">KV Cache Append</span>

<span style="font-size: 14px;">Every token an LLM generates during inference runs this kernel exactly once. The autoregressive decoder produces one new $(k, v)$ pair per step and must store them at position $t$ of a pre-allocated $(T_\text{max}, D)$ cache so the next attention call can read the whole history. The parallel pattern is the simplest in the entire curriculum: **a pure write-only 1D map**, one program per tile along the $D$ axis, no reads of the cache, no reductions, no accumulators. What makes this kernel matter is not its arithmetic but the regime it runs in. Decode-time inference launches it once per generated token, so the per-launch overhead dominates the per-byte work, and the right design choices are about minimizing fixed costs rather than maximizing throughput.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given new vectors $k_\text{new}, v_\text{new} \in \mathbb{R}^{D}$ and pre-allocated caches $K_\text{cache}, V_\text{cache} \in \mathbb{R}^{T_\text{max} \times D}$, the kernel writes</span>

$$
K_\text{cache}[\text{pos}, :] \leftarrow k_\text{new}, \quad V_\text{cache}[\text{pos}, :] \leftarrow v_\text{new}
$$

<span style="font-size: 14px;">In place: there is no separate output tensor. The caches are mutated, every other row left untouched, and the kernel returns nothing. $\text{pos}$ is a Python integer passed by value through the kernel signature, not a tensor. This is the unit operation that grows the KV cache by one row per generated token.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is one-dimensional: $\lceil D / \texttt{BLOCK\_SIZE} \rceil$ **programs**, each owning one contiguous tile of $D$ lanes. Program $p$ reads $\texttt{tl.program\_id(0)}$ and computes offsets $\texttt{offs} = p \cdot \texttt{BLOCK\_SIZE} + \texttt{tl.arange}(0, \texttt{BLOCK\_SIZE})$ into the $D$ axis. Each program does the same four operations: load its slice of $k_\text{new}$, load its slice of $v_\text{new}$, store the slice into $K_\text{cache}$ row $\text{pos}$, store the slice into $V_\text{cache}$ row $\text{pos}$.</span>

<span style="font-size: 14px;">For typical inference head dimensions ($D = 64$ to $D = 128$ per head, possibly $D = 4096$ after a head merge), one to four programs cover the whole vector at $\texttt{BLOCK\_SIZE} = 1024$. The grid is tiny by Triton standards. The independence is per-tile: programs never read each other's writes, and the kernel could run them in any order. There is no inter-program synchronization to worry about.</span>

<span style="font-size: 14px;">An alternative decomposition would parallelize across $K$ and $V$ ($2 \cdot \lceil D / \texttt{BLOCK\_SIZE} \rceil$ programs, each handling either $K$ or $V$ for a tile). The reference chooses the simpler fused-K-V form because the two writes share their offset arithmetic and the launch cost of one larger program beats the launch cost of two smaller programs at this scale. Both forms are correct; the fused form is slightly faster on the per-launch budget.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">The single constexpr meta-parameter is $\texttt{BLOCK\_SIZE} = 1024$. Power of two, large enough that the compiler emits wide vector instructions, small enough that the register footprint per program is trivial. The choice is the same as for vector add and for the same reason: there is no reuse, the kernel is bandwidth-bound, and the block size only needs to be wide enough to amortize the per-program launch cost.</span>

<span style="font-size: 14px;">The tail mask $\texttt{mask} = \texttt{offs} < D$ gates both loads and both stores. The mask is here for **correctness**, not performance: without it, when $D$ is not a multiple of $\texttt{BLOCK\_SIZE}$, the last program writes past the end of row $\text{pos}$ into the start of row $\text{pos} + 1$. Because the caches are pre-allocated and rows are densely packed, that out-of-range write corrupts an adjacent cache entry that the next decode step will read. The harness checks every row, not just row $\text{pos}$, so any cross-row write fails the test silently in the run but loudly in the result.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">There is no reuse anywhere in the kernel. Each input byte is read once; each output byte is written once. The four operations per program (two loads, two stores) operate on the same $\texttt{BLOCK\_SIZE}$ lanes and pass data straight through registers. No SRAM staging, no L2 reliance, no accumulator. The pointer arithmetic is the only computation: $k_\text{new\_ptr} + \texttt{offs}$ for the new vectors, $K_\text{cache\_ptr} + \text{pos} \cdot D + \texttt{offs}$ for the cache rows. The $\text{pos} \cdot D$ offset is computed once per program by the compiler-emitted prologue.</span>

<span style="font-size: 14px;">The cache itself is much larger than what one append touches. A typical KV cache is $(T_\text{max}, D)$ per head per layer, often hundreds of megabytes total across the model. The kernel writes $2 \cdot D \cdot 4$ bytes per call ($1$ KB for $D = 128$) into that cache, which is a thousand times smaller than what one fp32 matmul writes. The append's cost is dominated by launch overhead, not by these few hundred bytes of bandwidth.</span>

---

## <span style="font-size: 16px;">Launch Overhead, Not Bandwidth</span>

<span style="font-size: 16px;"></span>

<span style="font-size: 14px;">This is the operational angle that makes KV-append different from every other kernel in the curriculum. A kernel launch on a modern accelerator costs roughly $5$ to $10$ microseconds of fixed overhead, regardless of how small the kernel is. KV-append does $1$ KB of bandwidth at roughly $1$ TB/s, which is on the order of $1$ nanosecond of useful work. The arithmetic-to-overhead ratio is approximately $10^{-4}$. The kernel spends essentially all its wall-clock time on the launch itself, not on the writes.</span>

<span style="font-size: 14px;">During autoregressive decoding the model runs forward once per generated token. A single token's pass might launch dozens of small kernels (the QKV projection, the attention, the FFN, the layer norms, and KV-append per layer). Across a $32$-layer model, KV-append alone fires $64$ times per token (one per layer for $K$ and one for $V$, or fused into a single per-layer call as in this problem). At $5$ microseconds per launch, that is $0.3$ ms per token of pure launch overhead from KV-append calls, comparable to the actual compute on a fast accelerator.</span>

<span style="font-size: 14px;">The implication is that the right optimization is fusion, not inner-loop cleverness. Fusing KV-append into the QKV projection that produces $k_\text{new}$ and $v_\text{new}$ removes the round-trip through the output of the projection and the dedicated launch. Production inference engines (vLLM, TensorRT-LLM, sglang) do exactly this: the projection writes directly into the cache rows at $\text{pos}$ rather than into a temporary that a subsequent kernel scatters.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per output byte, the kernel does zero FLOPs. Arithmetic intensity is literally $0$ FLOPs per byte. The kernel is **as memory-bound as a kernel can be**: a pure scatter with no transformation between read and write. The roofline placement is at the leftmost edge of the bandwidth-bound region, and the only meaningful performance numbers are HBM bandwidth and launch overhead.</span>

<span style="font-size: 14px;">For the per-row write to actually saturate HBM bandwidth, $D$ would need to be in the millions of bytes; in real inference $D$ is in the hundreds. The kernel never gets close to the bandwidth ceiling because the work item is too small. This is fine for correctness and fine for inference latency once fused into the larger pipeline; standalone, it is a textbook example of why launch overhead is the dominant cost for tiny kernels.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author handles:** the 1D grid over $D$, the constexpr block size, the tail mask $\texttt{offs} < D$ on every load and store, the row-offset arithmetic $\text{pos} \cdot D$, the in-place semantic (no output tensor, just mutate the caches), and the per-call passing of $\text{pos}$ as a runtime Python int rather than a tensor or a constexpr.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering the two loads and two stores to wide PTX vector instructions, picking the warp count internally, and scheduling the two store pairs to issue concurrently. Because there is no arithmetic between load and store, the compiler has very little to schedule; the kernel is essentially a memcpy with a row offset, and the generated code reflects that. The author never names a warp, never declares scratchpad memory, never inserts a barrier.</span>

<span style="font-size: 14px;">The compiled output is small enough that the kernel's PTX is dominated by the function prologue and epilogue rather than by the inner work, which is another way of seeing the launch-overhead-dominated regime: even the per-instance setup of the program (computing pointers, deriving the mask) is comparable in cost to the actual memcpy.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">As a standalone kernel, the canonical implementation above is already optimal. The kernel does the minimum possible HBM traffic for the operation, masks the tail correctly, and fits comfortably in a single program for any reasonable $D$. Tuning $\texttt{BLOCK\_SIZE}$ between $256$ and $4096$ has negligible effect because the launch overhead dwarfs the bandwidth cost in all cases.</span>

<span style="font-size: 14px;">The real optimizations live outside the kernel boundary. **Fusion with the QKV projection** writes $k_\text{new}$ and $v_\text{new}$ directly into the cache rows at $\text{pos}$ as the projection's epilogue, eliminating both the temporary tensors and the dedicated launch. **CUDA-graph capture** records the sequence of kernel launches that compose one decode step (including this one) and replays the whole graph in a single submission, amortizing the launch overhead across the graph rather than paying it per kernel.</span>

<span style="font-size: 14px;">**Paged caches** generalize the layout: instead of a single $(T_\text{max}, D)$ buffer, the cache is split into fixed-size pages addressed by an indirection table. KV-append in that regime becomes a scatter into the page at $(\text{pos} // \texttt{page\_size}, \text{pos} \bmod \texttt{page\_size})$, plus a write to the indirection table on page boundaries. The kernel structure is the same, only the row-offset arithmetic changes.</span>

<span style="font-size: 14px;">**Prefill versus decode.** This kernel is the decode-time append, writing one new token. During prefill, the model processes the entire prompt at once and the equivalent operation writes many rows of the cache in a single launch. The kernel structure for prefill is a 2D grid over (token, $D$-tile) and reads from a $(T_\text{prefill}, D)$ block; the per-launch overhead becomes negligible because the work per launch is on the order of megabytes rather than kilobytes. The decode kernel and the prefill kernel are typically two separate $\texttt{@triton.jit}$ functions in production, even though their math is the same, because their performance regimes are different.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $T_\text{max} = 4$, $D = 6$, $\text{pos} = 2$, $\texttt{BLOCK\_SIZE} = 4$. The launch grid has $\lceil 6 / 4 \rceil = 2$ programs.</span>

<span style="font-size: 14px;">**Program 0** ($\texttt{pid} = 0$): offsets $\texttt{offs} = [0, 1, 2, 3]$, mask $\texttt{offs} < 6$ is $[T, T, T, T]$. Loads $k_\text{new}[0..3]$ and $v_\text{new}[0..3]$, stores them into $K_\text{cache}[2, 0..3]$ at base offset $2 \cdot 6 + 0 = 12$ and into $V_\text{cache}[2, 0..3]$ at the same offset structure.</span>

<span style="font-size: 14px;">**Program 1** ($\texttt{pid} = 1$): offsets $\texttt{offs} = [4, 5, 6, 7]$, mask $\texttt{offs} < 6$ is $[T, T, F, F]$. Lanes $0$ and $1$ load $k_\text{new}[4], k_\text{new}[5]$ and store into $K_\text{cache}[2, 4..5]$. Lanes $2$ and $3$ are masked off on the load and on the store. Critically: without the mask on the store, lanes $2$ and $3$ would write to $K_\text{cache}$ at offsets $14$ and $15$, which is row $2$ position $\{4, 5\}$ correctly for lanes $0, 1$, then row $2$ position $\{6, 7\}$ which is past the end of row $2$ and into the start of row $3$. Row $3$ would be silently corrupted.</span>

<span style="font-size: 14px;">Rows $0, 1, 3$ of both caches are never touched. The harness clones the caches before calling $\texttt{solve}$ and compares all $4$ rows; any cross-row write fails the test.</span>

<span style="font-size: 14px;">The same example with $D = 4$ (a multiple of $\texttt{BLOCK\_SIZE}$) needs only one program with a fully-true mask. The mask exists as compile-time scaffolding; when the runtime $D$ happens to be a multiple of the block size, every lane is in range and the mask is effectively a no-op. The kernel is correct in both cases without the author writing a special path for the aligned case, which is the standard advantage of the mask-and-overcommit pattern over an aligned-only kernel.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Missing tail mask on the stores.** The most common failure: the load is masked but the store is not, and the kernel writes past row $\text{pos}$ into row $\text{pos} + 1$. The masked-out lanes hold garbage from the input load and write it into the next cache entry, corrupting a row the harness explicitly checks.</span>
* <span style="font-size: 14px;">**Off-by-one in the row offset.** The base offset is $\text{pos} \cdot D$, not $(\text{pos} - 1) \cdot D$ or $\text{pos} \cdot (D - 1)$. The first error shifts the entire write up by one row; the second writes diagonally across rows and produces a smear pattern that is hard to debug.</span>
* <span style="font-size: 14px;">**Forgetting one of the two caches.** Writing only $K_\text{cache}$ and not $V_\text{cache}$ (or vice versa) passes a $K$-only or $V$-only check; the harness compares both and any partial implementation fails. Both stores must be in the kernel body.</span>
* <span style="font-size: 14px;">**Treating $\text{pos}$ as a tensor.** $\text{pos}$ is a Python int passed by value through the kernel argument list. Loading it as a tensor or passing it as a $\texttt{tl.constexpr}$ both break the kernel: the first reads garbage from a non-existent buffer, the second forces a recompile per call.</span>

---