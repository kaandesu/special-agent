# special-agent — The Big Plan

> GPU Programming course project (Politecnico di Torino, 2026/27).
> Written 2026-11-19, before any code exists. Update it as decisions change.
> Background and explanations: [`project-notes.md`](project-notes.md) and [`primers/`](primers/).
> Course rules: [`course/requirements.md`](course/requirements.md).

---

## 1. Goal

Train a **tiny language model** (~1M parameters) on my Mac and make it the brain of a **courier agent**. The agent lives in a text story world: doors, roads, bus stations, bus lines, streets, houses. That story world is a translation of a **deterministic network simulator**: opening a door brings an interface UP, a new bus line is a new route, knocking on the house is a ping. The agent never sees networking.

Then write **my own CUDA engine** that runs the trained agent, and study **where the agent's memory should live on the GPU**. That memory is the KV cache (the saved history of the episode). I build 4 versions (V0 to V3) and profile each one.

### The two research questions

| Side | Question | Measured by |
|---|---|---|
| **Agent** (A1–A4) | Can a tiny model, trained only on a simplified everyday abstraction, operate a more complex system correctly? Does the abstraction help compared with raw network text? Does it generalize to unseen scenarios? | Success rate on the **real network state**; story vs raw text; tokens per episode |
| **GPU** (V0–V3) | Where should the agent's memory (KV cache) and its environment live on the GPU? | Time per decision, decisions/sec, memory per agent, agents that fit, Nsight profiles |

**How they connect:** a simpler abstraction means fewer tokens per episode, which means a smaller KV cache, which means less GPU memory per agent and more agents fit (especially on the 2 GB Jetson).

### The one sentence for the professor

> "Training uses PyTorch; everything that runs the agent is our own CUDA, and the study is about where the agent's memory lives on the GPU."

---

## 2. Key decisions (already made)

| Decision | Choice |
|---|---|
| Training | **PyTorch, on my M1 Mac** (`device="mps"`). Training is not the new part. |
| Running the agent | **My own CUDA C engine**, no frameworks. This is the graded GPU work. |
| Correctness reference | PyTorch outputs for the same input. Not a CPU speed baseline. |
| Speed baseline | **V0 (naive GPU)**. Every other version is compared against it (and against PyTorch running the same model). |
| Environment | **Courier story world** on top of a **deterministic network state machine**, joined by a translator |
| Networking | Purely simulated. Real networking is future work. |
| Agent output | Never free text. At `ACT` it scores only the allowed option numbers (masked logits). |
| Tokenizer | Word-level, tiny vocabulary (~100–300 words), written by me |
| Hardware | Mac M1 16 GB (training) · RunPod NVIDIA GPU (CUDA + profiling) · Jetson Nano 2GB (edge) |
| Visualization | Stretch goal (CUDA–OpenGL interop) |

---

## 3. Repo layout

```
special-agent/
  docs/        PLAN.md (this file), project-notes.md, primers/, course/
  sim/         Python: network state machine, story translator, raw-text renderer, expert   (Phase 3)
  data/        data generator; outputs go to data/generated/ (git-ignored)                   (Phase 3)
  train/       Python/PyTorch: tokenizer, model, train, chat, play, export                    (Phases 1–4)
  cuda/        CUDA C engine: loader, kernels, agent loop V0–V3, Makefile                     (Phase 5)
  experiments/ profiling results and numbers, one folder per version/run                      (Phase 5)
```

---

## 4. Phases

Each phase has a goal, steps, the files it creates, a **"done when"** check, and the course topics it touches.

---

### Phase 0: Setup and learning (≈ 1–2 days)

**Goal:** a working Python environment and a basic understanding of how a GPT works.

Steps:
1. Install `uv` if missing: `brew install uv`.
2. Create the environment with **Python 3.12** (system Python is 3.15; PyTorch may not support it yet):
   ```bash
   cd ~/Desktop/Programming/gits/special-agent
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -r requirements.txt
   python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
   ```
3. Watch **Karpathy, "Let's build GPT: from scratch, in code, spelled out"** (~2 h). Pause and code along.
4. Skim **nanoGPT** (`model.py`, ~300 lines). Our model will be a smaller copy of its structure.
5. Read [`primers/01-llm-from-zero.md`](primers/01-llm-from-zero.md).

**Done when:** the last command prints `True` for MPS, and I can explain tokens → embedding → attention → scores in my own words.

---

### Phase 1: My first LLM, trained on the Mac (≈ 2–3 days)

**Goal:** see training, weights and text generation working end to end, on known data.

Data: **tiny Shakespeare** (~1 MB text file, the classic from Karpathy's video), character-level to start.

Files:
- `train/data_shakespeare.py`: download the text, build the character vocabulary, encode to IDs, 90/10 train/val split.
- `train/model.py`: tiny GPT (embedding + position embedding + N transformer blocks + unembedding). Reused in Phase 4.
- `train/train.py`: training loop (AdamW, batches of random windows, eval loss every N steps, save checkpoint).

Starting settings:

| Setting | Value |
|---|---|
| d_model (numbers per token) | 128 |
| layers | 4 |
| attention heads | 4 |
| context length | 256 for Shakespeare (512 later for episodes) |
| batch size | 32 |
| learning rate | 3e-4 |
| steps | ~5,000 (first test run: 200) |
| device | `mps` (fallback: `cpu`; set `PYTORCH_ENABLE_MPS_FALLBACK=1` if an op is missing) |

**Done when:**
- training loss and validation loss both go down and level off (char-level Shakespeare reaches roughly 1.5 or lower)
- a sample from the model looks Shakespeare-ish (made-up words and names in the right format)
- the checkpoint is saved in `checkpoints/` (git-ignored)

Course topics: none yet; this is the ML foundation.

---

### Phase 2: Talk to it (≈ 1 day)

**Goal:** interact with the trained model.

Files:
- `train/chat.py`: load the checkpoint, read a prompt from the keyboard, generate N tokens, print them, repeat.
  - options: `--temperature` (randomness), `--top_k` (only pick from the k most likely tokens), `--max_new_tokens`

Important expectation: this model **continues text, it does not chat**. Type `ROMEO:` and it writes a fake Romeo line. That's normal for a tiny model trained only on Shakespeare. Chat behavior comes from training on chat-shaped data, which is what Phase 4 does for the courier format.

**Done when:** I can type a prompt and watch it continue, and I understand how temperature and top-k change the output.

Also look at a few raw numbers: print the shape of every weight tensor and the parameter count. "The model is just float arrays" should feel concrete after this.

---

### Phase 3: The courier world (≈ 4–6 days)

**Goal:** a deterministic simulator that produces training data, in two text formats.

Full design: [`project-notes.md` §3](project-notes.md) (mapping table, example episode).

Files:
- `sim/network.py`: **network state machine**, the ground truth.
  - state: interfaces (UP/DOWN), links, routers, routing tables, destination host
  - operations: `link_up(if)`, `link_down(if)`, `route_add(subnet, via)`, `forward()`, `ping(host)`
  - `is_goal()`: destination reachable
- `sim/story.py`: **translator, network → story.** Describes the state as a scene + numbered options; maps a chosen option back to a network operation.

  | Story | Network |
  |---|---|
  | door A / B | eth0 / eth1 |
  | open / closed door | UP / DOWN |
  | road | link |
  | bus station | router |
  | timetable board | routing table |
  | bus line to a street | route to a subnet |
  | street, house number | subnet, host |
  | knock on the house | ping |

- `sim/raw.py`: **translator, network → raw network text** (`eth0 DOWN`, `route add 10.0.2.0/24 via r1`, …). Same states, same options, different words. Needed for experiment A2.
- `sim/expert.py`: BFS over states → the shortest correct action sequence.
- `sim/scenarios.py`: random scenario generator, with difficulty knobs:
  - number of doors, stations, streets, houses
  - multi-hop routes (change buses)
  - dead ends, distracting options
  - stretch: locked gate = firewall, address book = DNS
- `data/gen_data.py`: run the expert on N random scenarios and write episodes to `data/generated/`:
  - `story_train.txt`, `story_test_seen.txt`, `story_test_unseen.txt`
  - the same three for `raw_`
  - **"unseen" = scenario types held out from training** (e.g. bigger towns, more hops), not just new random seeds
- `train/tokenizer.py`: word-level vocabulary built from all the words the translators can print, plus special tokens `<s>`, `SEE`, `OPTIONS`, `ACT`, `RESULT`, `DONE`. Saved to `data/generated/vocab.json`.

Episode format (story):
```
<s> you are at the depot . parcel for house 5 on maple street .
SEE door A closed | door B closed
OPTIONS 1 open door A 2 open door B 3 wait
ACT 1
SEE door A open . road A leads to north station
...
RESULT parcel delivered DONE
```

Size: start with ~20,000 train episodes, scale to ~50,000 if needed (~5–15M tokens).

**Done when:**
- I can play the simulator by hand in the terminal (a tiny `sim/play_human.py`) and every option does what the mapping says
- the expert solves 100% of generated scenarios
- story and raw files describe exactly the same episodes (same scenario seeds)
- I know tokens per episode for story vs raw (first A2 number)

Course topics: none yet, but the transition table design here becomes `__constant__` memory in Phase 5. Keep it as a flat integer table: `next_state[state][option]`.

---

### Phase 4: Train the courier agent (≈ 3–5 days)

**Goal:** an agent that delivers parcels in scenarios it has never seen, and a weights file for the CUDA engine.

Files:
- `train/train_agent.py`: same model as Phase 1, trained on `story_train.txt`. Context 512, vocab from `tokenizer.py`.
  - loss: mainly on the token right after each `ACT` (that's the decision); optionally a small weight on the other tokens
- `train/play.py`: the agent loop in Python.
  - feed the episode so far → scores at `ACT` → **mask to allowed options** → pick the best → `sim` step → append → repeat
  - after the episode: check the **network** state (`is_goal()`), not only the story text
  - report the success rate on `test_seen` and `test_unseen`
- `train/export.py`: write `checkpoints/agent.bin` for the CUDA engine
- `train/reference.py`: for a fixed input, save **every intermediate output** (after embedding, each layernorm, attention, MLP, final scores) to `checkpoints/reference/`. This is how the CUDA kernels get checked in Phase 5.

`agent.bin` format (keep it boring):
```
header:  magic "SAGT" | version (int32) | vocab_size | d_model | n_layers | n_heads | ctx_len   (int32 each)
tensors: float32, little-endian, row-major, in a FIXED documented order:
         tok_emb, pos_emb,
         for each layer: ln1_w, ln1_b, Wq, Wk, Wv, Wo, ln2_w, ln2_b, W1, b1, W2, b2,
         lnf_w, lnf_b   (unembedding tied to tok_emb)
```
(If the model uses one fused `Wqkv` matrix, export it split, or document the layout. Write the order in `train/README.md` and keep it the same in `cuda/weights.cu`.)

Experiments:
- **A1:** success rate on `test_seen`, checked on the network state
- **A3:** success rate on `test_unseen` (bigger or different towns). This is the real generalization test.
- **A2:** train a second agent on `raw_train.txt` (same size, same steps). Compare success rate, steps to learn, tokens per episode.

**Done when:**
- success rate on `test_seen` is high (aim ≥ 95%)
- the `test_unseen` number is measured and written down, whatever it is
- the A2 comparison table exists
- `agent.bin` + `reference/` exist and load back correctly in Python

---

### Phase 5: The GPU project (RunPod + Jetson) (≈ 3–4 weeks, the main part)

**Goal:** my own CUDA engine runs the courier agent, in 4 memory versions, all profiled.

Read first: [`primers/03-agent-memory.md`](primers/03-agent-memory.md), [`primers/04-cuda-in-this-project.md`](primers/04-cuda-in-this-project.md), [`primers/05-runpod-and-jetson.md`](primers/05-runpod-and-jetson.md).

#### 5.1 RunPod setup (do this early, even during Phase 3)
1. Account, ~$10 credit, cheapest NVIDIA pod with the official **RunPod PyTorch** template (includes CUDA + `nvcc`).
2. `nvidia-smi`, `nvcc --version`.
3. **Test `ncu` on a toy kernel.** If it fails with `ERR_NVGPUCTRPERM`, counters are blocked: try another pod type or provider, or plan around Nsight Systems + manual timing.
4. **Stop the pod when not using it.**

#### 5.2 Engine skeleton
Files:
- `cuda/weights.cu`: read `agent.bin` → `cudaMalloc` + `cudaMemcpy`
- `cuda/sim.cu` + `cuda/sim_table.h`: the state machine as integer tables (exported from Python), and the story text per state as pre-made token lists
- `cuda/kernels/`: `embed.cu`, `matmul.cu`, `layernorm.cu`, `attention.cu`, `softmax.cu`, `gelu.cu`, `pick_action.cu`
- `cuda/main.cu`: command-line entry: `--version v0|v1|v2|v3`, `--agents N`, `--check`, `--bench`
- `cuda/Makefile`: `nvcc -O3 -lineinfo`, plus an `-arch` setting per machine (`sm_53` for the Jetson, the right one for the RunPod GPU)

#### 5.3 Correctness first
- `./agent --check`: run the fixed input from Phase 4 and compare every intermediate output with `checkpoints/reference/` (max absolute difference, e.g. < 1e-4 at FP32).
- Then play the test scenarios: the success rate must match the Python `play.py` result.
- Debug wrong kernels with `cuda-gdb` / `compute-sanitizer` (Lab 7).

#### 5.4 The four versions

| Version | Change | Expected result |
|---|---|---|
| **V0, naive** | recompute the whole history every step; simulator on the CPU; copy scores to the CPU each step | time per decision grows with the episode; timeline full of copies and CPU gaps |
| **V1, KV cache in global memory** | process only new tokens; append K,V to arrays in global memory | time per decision almost flat |
| **V2, whole loop on the GPU** | `sim_step` kernel; transition table in `__constant__`; no CPU between steps (try CUDA Graphs) | the gaps disappear from the timeline |
| **V3, KV cache in shared memory** | persistent kernel, one block per agent; cache in shared memory (with design choices: FP16, recent-window in shared + rest in global, one layer at a time) | lower latency per agent, but lower occupancy: **measure the trade-off** |

Then the multi-agent runs: 1, 10, 100, 1,000+ agents in parallel (batching, streams).

#### 5.5 Kernel optimization (in parallel with 5.4)
- matmul: naive → tiled in shared memory → compare with **cuBLAS** (lecture 5, Lab 4, libraries)
- softmax / layernorm: shared-memory reduction → **warp shuffle** (`__shfl_down_sync`) (lectures 6, 9)
- `sim_step`: `__constant__` vs shared vs `__ldg` vs global for the transition table, with agents in mixed states (broadcast vs serialized reads); measure **divergence**
- **atomics**: count actions taken / parcels delivered across all agents (histogram, lectures 6 and 8)
- FP16 weights and KV cache (half the memory; matters most on the Jetson)

#### 5.6 Profiling workflow (every version, every kernel change)
```bash
make && ./agent --check                    # correct?
./agent --bench --version v1 --agents 1    # ms per decision, decisions/sec
nsys profile -o experiments/v1/timeline ./agent --version v1
ncu --set full -k attention -o experiments/v1/attention ./agent --version v1
# on the Jetson: nvprof --print-gpu-trace ./agent ...   (ncu doesn't support Maxwell)
```
Write every result into `experiments/<version>/NOTES.md`: setup, numbers, a screenshot, what I learned, what I change next.

#### 5.7 Jetson Nano 2GB
- Build with `-arch=sm_53` on CUDA 10.2 (avoid newer CUDA features, or guard them with `#if`).
- Copy `agent.bin` over. Run V1–V3 and measure latency per decision, memory per agent, **maximum agents that fit**, FP32 vs FP16.
- Profile with `nvprof`.

**Done when:**
- all four versions pass `--check` and match Python's success rate
- each version has numbers plus profiler evidence in `experiments/`
- the Jetson vs RunPod comparison exists
- the graphs for the report can be drawn (see Phase 6)

Course topics: almost all of them. See [`course/topics.md`](course/topics.md).

---

### Phase 6: Deliverables

#### 6.1 Proposal (for the Nov 25–26 quick review)
- 1-page PDF in the style of the examples in [`course/example-proposals/`](course/example-proposals/): title, problem statement, implementation, KPIs, hardware, references.
- Draft text: [`project-notes.md` §5 "WIP Proposal"](project-notes.md).
- 1–2 slides and a **3-minute** spoken pitch.
- Ask the professor the open questions (§7).

#### 6.2 Report (written manuscript)
1. Problem and motivation
2. Background: transformer forward pass, KV cache, GPU memory hierarchy
3. The courier world and the network abstraction
4. Training setup (short: PyTorch, data, model size)
5. The CUDA engine: kernels and design
6. V0–V3: design, profiling evidence, numbers
7. Jetson vs RunPod
8. Agent results: A1, A2, A3
9. Limitations and future work (real networking, swapping translators, pretrained models)

#### 6.3 Graphs to produce
1. Time per decision vs step number, V0–V3 (V0 rises, the others stay flat)
2. Decisions per second, V0–V3, at 1 and 1,000 agents
3. Memory per agent and max agents that fit: Jetson vs RunPod, FP32 vs FP16
4. Roofline position of the main kernels (attention, matmul) for V1 vs V3
5. Nsight Systems timeline screenshots: V0 vs V2
6. My matmul vs cuBLAS
7. Story vs raw text: success rate and tokens per episode

#### 6.4 Presentation (30% of the grade)
- slides + a live demo: the courier delivering a parcel, with the network state shown next to it
- if visualization was built: the live town view on the Jetson

---

### Stretch goals (only after Phase 5 is done)
- CUDA–OpenGL interop: couriers moving through the town map live; attention heatmaps (which earlier words the agent looked at)
- paged KV cache for many episodes of different lengths
- harder worlds: firewalls (locked gates), DNS (address book), broken links (closed roads)

---

## 5. Timeline

| When | Phase | Milestone |
|---|---|---|
| Nov 19–21 | 0 | environment works, Karpathy video watched |
| Nov 21–24 | 1–2 | Shakespeare model trained on the Mac, `chat.py` works; RunPod `ncu` test; proposal PDF + slides |
| **Nov 25–26** | 6.1 | **Quick review: pitch, get the professor's OK** |
| Nov 27 – Dec 5 | 3 | courier world, expert, data, tokenizer |
| Dec 5–12 | 4 | courier agent trained; A1/A2/A3 first numbers; `agent.bin` + reference outputs |
| Dec 12–16 | 5 | CUDA engine passes `--check`; V0 + V1 numbers |
| **Dec 16–17** | | **Verification: show the agent delivering + V0 vs V1** |
| Dec 18 – Jan 6 | 5 | V2, V3, kernel optimization, multi-agent, Jetson port, full profiling |
| **Jan 7–8** | | **Final check-up** |
| Jan 8 → exam date | 6 | report, slides, demo (+ visualization if time allows) |
| Exam session Jan 11 – Feb 20, 2027 | | book the exam, deliver the project by the exam date |

Feedback in **at least one** verification session is mandatory.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `ncu` blocked on RunPod (performance counters) | test in week 1; other pod type/provider; fall back to Nsight Systems + manual timing |
| Jetson stuck on CUDA 10.2 / SM 5.3 | keep engine code plain CUDA C; `#if` guards; Makefile target per machine |
| Learning LLMs from zero takes longer than planned | Phases 1–2 are small on purpose; ask Claude to explain; the model stays tiny |
| CUDA kernels give subtly wrong numbers | `reference/` dumps from PyTorch, `--check` mode, `compute-sanitizer` |
| Agent doesn't generalize (A3 low) | that's a valid result to report; try more scenario variety, longer training |
| Story vs raw shows no difference (A2) | also a valid, honest result; still report tokens per episode → memory |
| V3 doesn't fit in shared memory (especially on the Jetson) | that's the trade-off being studied: FP16, shorter window, one layer at a time |
| Scope creep | visualization and paged cache only after V0–V3 work |
| MPS missing an operation | `PYTORCH_ENABLE_MPS_FALLBACK=1`, or train on CPU (still fast at this size) |

---

## 7. Open questions

- [ ] Solo or group? Who? (Max 3. Possible split: training + matmul/attention · world + agent loop V0–V3 · Jetson + profiling + visualization)
- [ ] Is it OK with the professor that training uses PyTorch and the graded GPU work is the CUDA engine + memory study?
- [ ] Is the Jetson Nano 2GB flashed and working (JetPack 4.6, `nvcc`, `nvprof`)?
- [ ] Which RunPod GPU? Does `ncu` work there?
- [ ] Can I use the ACS lab GPU machines as well?
- [ ] Visualization: stretch goal or cut?
- [ ] Final world size and which extras (firewall, DNS)?

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **Token** | one unit of text the model reads (here: a word or a character), stored as an integer ID |
| **Vocabulary** | the full list of tokens the model knows |
| **Weights / parameters** | the float arrays inside the model; training sets their values |
| **Embedding** | the row of 128 numbers that represents a token |
| **Forward pass** | running the model once: tokens in → scores out |
| **Logits / scores** | the model's raw output, one number per vocabulary word; higher = more likely next |
| **Softmax** | turns scores into probabilities that sum to 1 |
| **Loss** | how wrong the model was; training makes it go down |
| **Backward pass / gradient** | how much each weight should change to reduce the loss (PyTorch computes it) |
| **AdamW** | the standard rule for updating weights from gradients |
| **Attention** | each token looks back at earlier tokens and mixes in the relevant ones (Q, K, V) |
| **KV cache** | saved K and V vectors of earlier tokens, so they're not recomputed: **the agent's memory** |
| **Context length** | max number of tokens the model can look at (512 here) |
| **Masked options** | ignoring every score except the allowed option numbers |
| **Behavior cloning** | training the model to copy an expert's actions |
| **MPS** | Apple's GPU backend for PyTorch on the Mac |
| **Kernel** | a function that runs on the GPU, in many threads at once |
| **SM** | streaming multiprocessor: one of the GPU's "cores", which runs blocks |
| **Warp** | 32 threads that execute together |
| **Shared memory** | small, fast memory per SM, shared by one block |
| **Global memory** | the GPU's big, slow main memory (GBs) |
| **Constant memory** | small read-only memory, fast when a warp reads the same address |
| **Occupancy** | how many warps are active on an SM compared with the maximum |
| **Divergence** | threads in one warp taking different branches, which then run one after the other |
| **Coalescing** | neighboring threads reading neighboring addresses, so reads combine into few transactions |
| **Roofline** | a chart showing whether a kernel is limited by compute or by memory bandwidth |
| **Persistent kernel** | a kernel that keeps running and loops over work, instead of being relaunched |
| **Nsight Systems / Compute** | NVIDIA profilers: the whole-program timeline / single-kernel detail |
| **nvprof** | the older NVIDIA profiler; the one that works on the Jetson Nano |
