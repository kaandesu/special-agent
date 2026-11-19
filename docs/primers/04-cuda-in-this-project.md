# Primer 4 — Where CUDA actually happens in this project

**Yes, we write `.cu` files, and yes, we profile them.** PyTorch does exactly one job: training the model and saving `agent.bin`. Everything after that is my own CUDA code, and that's the graded part.

## Where the `.cu` files are

```
cuda/
  main.cu         load agent.bin, run the agent loop, print timings (--version, --agents, --check, --bench)
  weights.cu      read agent.bin → cudaMalloc + cudaMemcpy onto the GPU
  sim.cu          the courier world / network state machine, on the GPU
  sim_table.h     the transition table, exported from Python
  kernels/
    embed.cu      token ID → its embedding row
    matmul.cu     x · W           (naive first, then tiled in shared memory)
    softmax.cu    scores → probabilities   (reduction, warp shuffle)
    layernorm.cu  rescale vectors          (reduction)
    attention.cu  q · K, softmax, · V, reading the KV cache
    gelu.cu       elementwise
    pick_action.cu  masked argmax over the allowed options
  v0_naive.cu     the agent loop, versions V0..V3
  v1_kvcache.cu
  v2_ongpu.cu
  v3_shared.cu
  Makefile        nvcc ...
```

## What a kernel looks like

The most basic one: multiply one token's vector by a weight matrix, with each thread computing one output number (as in Lab 2):

```cuda
// out[j] = sum over i of x[i] * W[i][j]       x: [128], W: [128 × 512], out: [512]
__global__ void matvec(const float* x, const float* W, float* out, int n_in, int n_out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;     // which output this thread computes
    if (j >= n_out) return;
    float sum = 0.0f;
    for (int i = 0; i < n_in; i++)
        sum += x[i] * W[i * n_out + j];
    out[j] = sum;
}
// launch: matvec<<<(512 + 127) / 128, 128>>>(d_x, d_W1, d_h, 128, 512);
```

Then you improve it, as in Lab 4: load tiles into shared memory, make sure reads are coalesced, and so on.

The simulator on the GPU, one thread per agent (V2):

```cuda
__constant__ int next_state[MAX_STATES][MAX_OPTIONS];   // the state machine table

__global__ void sim_step(int* state, const int* action, int* done, int n_agents) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;      // which agent
    if (a >= n_agents || done[a]) return;
    state[a] = next_state[state[a]][action[a]];         // "open door A" → eth0 UP
    done[a]  = (state[a] == GOAL_STATE);
}
```

And `main.cu` stitches the kernels into the agent loop:

```cuda
load_weights("agent.bin");                     // cudaMemcpy to the GPU
for (step = 0; step < max_steps; step++) {
    embed<<<...>>>(...);
    for (l = 0; l < 4; l++) {
        layernorm<<<...>>>(...);
        matmul<<<...>>>(...);   attention<<<...>>>(...);
        layernorm<<<...>>>(...);
        matmul<<<...>>>(...);   gelu<<<...>>>(...);   matmul<<<...>>>(...);
    }
    pick_action<<<...>>>(...);
    sim_step<<<...>>>(...);                    // V2+: no CPU involved
}
```

## How each course activity shows up

| Course | In this project |
|---|---|
| Lab 2: first kernels, grids/blocks/indexing | `matvec`, `gelu`, `sim_step`: one thread per number or per agent |
| Lecture 4: divergence, occupancy | agents in different states take different branches in `sim_step`; block size choices |
| Lecture 5 + Lab 4: shared/constant memory, tiling, coalescing | tiled matmul; transition table in `__constant__`; **V3 KV cache in shared memory** |
| Lecture 6: atomics, reductions | softmax/layernorm sums; counting delivered parcels or actions taken across agents |
| Lecture 9: warp shuffle, cooperative groups | fast softmax/layernorm sums with `__shfl_down_sync` |
| Lecture 7 + Lab 5: profiling, roofline | **every version is measured** with `nsys`, `ncu`, or `nvprof` on the Jetson |
| Lecture 8 + Lab 6: streams | run many agent groups at once; overlap copies with compute |
| Lab 7: cuda-gdb | debugging kernels when the scores come out wrong |
| Week 11: libraries | compare my matmul against cuBLAS |

## The workflow, repeated for every kernel and version

```bash
nvcc -O3 -o agent cuda/main.cu ...          # 1. compile
./agent --check                              # 2. correctness: do my outputs match PyTorch's reference dumps?
./agent --bench                              # 3. speed: decisions/sec, ms per decision
nsys profile ./agent                         # 4. timeline: where are the gaps and copies?
ncu --kernel-name attention ./agent          # 5. kernel detail: bandwidth, occupancy, roofline
                                             # 6. optimize → back to 1, and write the numbers down
```

The report is basically the story of that loop: *"V0 took X ms per decision. Nsight showed the time went to copies, so V2 moved the simulator onto the GPU. Attention was memory-bound, so V3 put the cache in shared memory. Here are the numbers."*

## Profiling ≠ the agent's mind
- **Profilers** measure **how the GPU runs**: time per kernel, bandwidth, occupancy, stalls.
- **What the agent "thinks"** (attention weights, activations) is data you copy out of GPU memory and **visualize** separately.
- Profiling answers "is it fast and why?" Visualization answers "what is it doing?"

All of this happens on RunPod, the Jetson or a lab machine. The Mac can't compile or run `.cu` files.
