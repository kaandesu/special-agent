# Primer 5 — RunPod and the Jetson Nano

## RunPod (NVIDIA GPU in the cloud)

Used for: building and running the CUDA engine, profiling, multi-agent runs. Not needed for training (the Mac does that).

### Setup
1. Make an account at runpod.io and add about **$10** of credit. That's plenty for this project.
2. **Deploy a pod:**
   - a cheap GPU (an RTX 3090/4090-class card from "Community Cloud" is usually well under $1/hour; check current prices)
   - the official **RunPod PyTorch** template, which already includes the CUDA toolkit and `nvcc`
3. Open the web terminal or connect over SSH.
4. `git clone https://github.com/kaandesu/special-agent.git`
5. Check:
   ```bash
   nvidia-smi          # which GPU, driver version
   nvcc --version      # CUDA toolkit version
   ```
6. Copy `agent.bin` and `checkpoints/reference/` over (they're git-ignored): `scp`, or `runpodctl send`.
7. **Stop the pod when you're done.** It bills every hour it's running, even if you're not using it.

### Risk: profiling may be blocked
- Rented containers often **block GPU performance counters**. Nsight Compute (`ncu`) then fails with `ERR_NVGPUCTRPERM`. You can't change the host driver setting from inside a container.
- One of the example proposals (Angry Santa) hit exactly this.
- **Test this in week 1**, with a toy kernel:
  ```bash
  ncu ./toy_kernel
  ```
- If it fails: try another pod type (e.g. a "Secure Cloud" pod) or another provider; ask whether the ACS lab machines allow it; otherwise use Nsight Systems (`nsys`, usually works without counters) plus manual timing with CUDA events.

## Jetson Nano 2GB (the edge device)

| Spec | Value |
|---|---|
| GPU | Maxwell, **SM 5.3**, 128 CUDA cores |
| Memory | **2 GB, shared between CPU and GPU** (the OS takes a big part) |
| Shared memory | 48 KB per block, 64 KB per SM |
| Software | stuck on **JetPack 4.6 / CUDA 10.2 / Ubuntu 18.04** |
| Profiler | **`nvprof`** (Nsight Compute doesn't support Maxwell) |
| FP16 | supported; worth testing for weights and KV cache |

What it's good for:
- **inference** and edge measurements: latency per decision, memory per agent, **maximum agents that fit in 2 GB**
- the V3 shared-memory trade-off is extreme here (48 KB per block), which makes it a good contrast to the big GPU
- has a display output, so the CUDA–OpenGL visualization (stretch goal) can run live

What it's bad for:
- training (and modern PyTorch doesn't run on it anyway), so train on the Mac and copy `agent.bin` over

Code rules so the engine builds on both machines:
- plain CUDA C, no features newer than CUDA 10.2, or guard them with `#if CUDART_VERSION >= ...`
- Makefile target per machine: `-arch=sm_53` for the Jetson, the right `sm_XX` for the RunPod GPU
- profile with `nvprof --print-gpu-trace ./agent ...`, or `nvprof --metrics achieved_occupancy,gld_efficiency ./agent ...`

## The lab machines
The course labs (ACS Lab, DAUIN dept., first floor) have GPU development cards. Ask whether they can be used for the project, and whether profiling counters work there.
