# special-agent

**Where should an agent remember?** A tiny language model acting as a courier agent, and a custom CUDA engine that studies where the agent's memory should live on the GPU.

GPU Programming course project, Politecnico di Torino, 2026/27.

## The idea in 30 seconds

- A **tiny transformer** (~1M parameters) is trained in **PyTorch on a Mac**.
- It plays a **courier** in a text story world: open doors, walk roads, take buses, deliver the parcel to the house.
- That story world is a translation of a **deterministic network simulator**: a door is a network interface (open = UP), a bus station is a router, a bus line is a route, the house is the destination server. The agent never sees networking.
- **Agent question:** can a tiny model operate a more complex system correctly through a simple everyday abstraction, and does the abstraction help?
- **GPU question:** a **custom CUDA engine** runs the trained agent in 4 versions (V0–V3) that move its memory (the KV cache) and its simulator closer to the GPU cores, profiled with Nsight on RunPod and on a Jetson Nano 2GB.

> Training uses PyTorch; everything that runs the agent is our own CUDA.

## Status

Planning done, no code yet. Next: **Phase 0** (environment + learning).

## Start here

| Read | For |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | **The big plan**: phases 0–6, steps, "done when", timeline, risks, glossary |
| [`docs/project-notes.md`](docs/project-notes.md) | Full background: course rules, the story world and mapping, research questions, LLM primer, WIP proposal |
| [`docs/primers/`](docs/primers/) | Short explainers: [LLMs from zero](docs/primers/01-llm-from-zero.md) · [Mac vs CUDA](docs/primers/02-mac-vs-cuda.md) · [agent memory](docs/primers/03-agent-memory.md) · [CUDA in this project](docs/primers/04-cuda-in-this-project.md) · [RunPod & Jetson](docs/primers/05-runpod-and-jetson.md) |
| [`docs/course/`](docs/course/) | Course [guide](docs/course/guide.md), [requirements and dates](docs/course/requirements.md), [syllabus → project mapping](docs/course/topics.md), [example proposals](docs/course/example-proposals/notes.md) |
| [`docs/original-idea.md`](docs/original-idea.md) | The first brainstorm |

## Phases

| # | Phase | Where |
|---|---|---|
| 0 | Setup + learning (Python 3.12 venv, PyTorch/MPS, Karpathy's "Let's build GPT") | Mac |
| 1 | First LLM: tiny Shakespeare, trained on the Mac | Mac |
| 2 | Talk to it (`chat.py`) | Mac |
| 3 | The courier world: network state machine, story + raw translators, expert, data | Mac |
| 4 | Train the courier agent, measure A1–A3, export `agent.bin` | Mac |
| 5 | **The GPU project:** CUDA engine, V0–V3, profiling, Jetson | RunPod / Jetson |
| 6 | Proposal, report, slides, demo | — |

## Layout

```
docs/          plan, notes, primers, course material
sim/           courier world + network state machine (Python)
data/          data generator; outputs in data/generated/ (git-ignored)
train/         PyTorch: model, training, chat, play, export
cuda/          the CUDA agent engine
experiments/   profiling results and numbers
```

## Hardware

- **Apple M1, 16 GB**: training (PyTorch, MPS)
- **RunPod NVIDIA GPU**: CUDA engine, Nsight Systems / Compute
- **Jetson Nano 2GB** (Maxwell SM 5.3, CUDA 10.2): edge runs, `nvprof`
