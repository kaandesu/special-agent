# Primer 2 — Training on the Mac vs on CUDA

## My machine
- Apple **M1**, **16 GB** memory (shared between CPU and GPU)
- System Python is **3.15**, which is very new; PyTorch may not support it yet → use a **Python 3.12** venv (`.python-version` in the repo root)

## Can I train on the Mac? Yes.
- The model is ~1M parameters, which is tiny. Big LLMs have billions.
- PyTorch uses the M1's GPU through Apple's **MPS** backend: `device = "mps"`.
- Estimated time: roughly **10–30 minutes** for a full run (5,000 steps, batch 32). First test runs: 1–2 minutes.
- Fallback: `device = "cpu"` still works at this size, just slower.

## What's different from running on an NVIDIA GPU (besides speed)?

For training this model: **nothing important. It's the same math, so you get an equally good model.**

Small differences:
- **Tiny number differences.** GPUs add floats in slightly different orders, so weights won't match bit-for-bit across machines (or even across two runs). Behavior is the same.
- **Some features are NVIDIA-only:** the fastest attention implementations, `torch.compile` speed-ups, good bfloat16 support (the M1 handles it poorly). None of these matter at ~1M parameters.
- **Rare missing operations on MPS:** set `PYTORCH_ENABLE_MPS_FALLBACK=1` so they fall back to the CPU. A basic GPT doesn't hit this.
- **Memory:** 16 GB shared, like the Jetson. Far more than needed.

## The weights file is portable
Train on the Mac → `agent.bin` → load it unchanged on RunPod or the Jetson. A float32 is the same 4 bytes on all three, and all three are little-endian.

## Where the difference really matters: the course
Everything that gets graded only exists on NVIDIA hardware:
- writing and running `.cu` files (the Mac **can't run CUDA at all**)
- the agent memory experiments (V0–V3)
- the profilers: Nsight Systems, Nsight Compute, nvprof

**So: train and play on the Mac; build, run and profile the CUDA engine on RunPod / the Jetson / the lab machines.**
