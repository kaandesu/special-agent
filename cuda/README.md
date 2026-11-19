# cuda/ — the agent engine, my own CUDA C (Phase 5)

Runs on an NVIDIA GPU only (RunPod, Jetson Nano, lab machines). The Mac can't build this.

Planned layout:
```
main.cu          --version v0|v1|v2|v3  --agents N  --check  --bench
weights.cu       agent.bin → cudaMalloc + cudaMemcpy
sim.cu           courier world / network state machine on the GPU (V2+)
sim_table.h      transition table exported from Python (__constant__)
kernels/         embed, matmul, layernorm, attention, softmax, gelu, pick_action
v0_naive.cu      recompute everything; simulator on the CPU
v1_kvcache.cu    KV cache in global memory
v2_ongpu.cu      whole loop on the GPU
v3_shared.cu     KV cache in shared memory, persistent kernel, one block per agent
Makefile         nvcc -O3 -lineinfo; -arch=sm_53 for the Jetson, sm_XX for RunPod
```

Rules:
- `--check` must pass (outputs match `checkpoints/reference/` from PyTorch) before any benchmark counts.
- Stay compatible with CUDA 10.2 (Jetson), or guard newer features with `#if`.

Read first: [`../docs/primers/03-agent-memory.md`](../docs/primers/03-agent-memory.md), [`../docs/primers/04-cuda-in-this-project.md`](../docs/primers/04-cuda-in-this-project.md).
