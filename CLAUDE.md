# CLAUDE.md — context for future sessions

## What this repo is
GPU Programming course project (Politecnico di Torino, 2026/27). A tiny transformer (~1M params) is trained in PyTorch on the user's Mac and acts as a courier agent in a text story world, which is an abstraction of a deterministic network state machine. A custom CUDA engine runs the trained agent and studies where the agent's memory (KV cache) and its simulator should live on the GPU (versions V0–V3), measured by profiling.

The plan is [`docs/PLAN.md`](docs/PLAN.md). Read it before doing anything. Background is in [`docs/project-notes.md`](docs/project-notes.md).

## The user
- Starting from **zero knowledge of LLMs**. Explain ML concepts plainly, expand acronyms, show concrete shapes and numbers, prefer line-by-line traces of what the code does.
- Knows C and is taking the GPU course (CUDA, memory hierarchy, profiling labs).
- Wants to understand the code, not just receive it. Build in small steps, phase by phase; explain each new file.

## Decisions (don't re-litigate without the user)
- **PyTorch for training only.** Everything that runs the agent is hand-written **CUDA C** (the graded part).
- No CPU speed baseline. **V0 (naive GPU) is the baseline.** PyTorch outputs are the correctness reference.
- Story world (doors = interfaces, bus stations = routers, bus lines = routes, houses = hosts) over a network state machine; networking is purely simulated.
- The agent never generates free text: at `ACT` it scores only the allowed option numbers (masked logits).
- Word-level custom tokenizer, tiny vocabulary.
- Visualization (CUDA–OpenGL) is a stretch goal.

## Hardware and environment
- **Mac M1 16 GB**: training. System Python is 3.15 → use a **Python 3.12** venv (`.python-version`), `device="mps"`, fallback `PYTORCH_ENABLE_MPS_FALLBACK=1`. **The Mac cannot build or run CUDA.**
- **RunPod NVIDIA GPU**: CUDA engine and Nsight. `ncu` may be blocked (`ERR_NVGPUCTRPERM`); test early.
- **Jetson Nano 2GB**: Maxwell SM 5.3, JetPack 4.6 / CUDA 10.2, `nvprof` only (no Nsight Compute). Keep engine code CUDA-10.2-compatible or guard with `#if`.

## Conventions
- Generated data → `data/generated/`; checkpoints/weights → `checkpoints/`; both git-ignored.
- `agent.bin` format is specified in `train/README.md`; `cuda/weights.cu` must match it exactly.
- Every benchmark needs `--check` passing first. Results go in `experiments/<name>/NOTES.md`.
- Course example proposal PDFs in `docs/course/example-proposals/` are git-ignored (other students' work; the repo is public).

## Key dates
Nov 25–26, 2026 quick review (3-min pitch, 1–2 slides) · Dec 16–17 verification · Jan 7–8, 2027 final check-up · exam session Jan 11 – Feb 20, 2027.

## Git
- **Never add a `Co-Authored-By: Claude` line (or any Claude/AI attribution) to commits or PRs.** All commits are authored by the user only.
