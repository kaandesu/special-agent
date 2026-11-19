# experiments/ — measurements (Phases 4 and 5)

One folder per version or experiment, e.g. `v0/`, `v1/`, `v3-fp16/`, `jetson-v2/`, `a2-story-vs-raw/`.

Each folder has a `NOTES.md` with:
- **setup:** machine, GPU, CUDA version, commit hash, command line
- **numbers:** ms per decision, decisions/sec, memory per agent, agents that fit, success rate
- **evidence:** profiler screenshots or exported summaries
- **what I learned** and **what I change next**

Raw profiler reports (`*.nsys-rep`, `*.ncu-rep`) are git-ignored because they're large. Keep screenshots and summaries here instead.
