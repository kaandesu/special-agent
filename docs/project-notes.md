# GPU Programming Project — Personal Planning Notes

> Personal notes for me. This is NOT the proposal. The 1-page proposal PDF is a short, polished version of section 5.
> Original brainstorm: `original-idea.md`. The step-by-step plan is `PLAN.md`.
> Last updated: 2026-11-19.

---

## 0. TL;DR

- **Project:** a tiny transformer, trained in PyTorch on my Mac and then run by my own CUDA engine (no frameworks at run time). It acts as the "brain" of a **delivery agent** in a text-based story world (doors, roads, bus stations, houses).
- **Behind the scenes, the story world is a network simulator.** Opening a door sets an interface UP; asking the bus station for a new line adds a route; knocking on the house is a ping. The agent never sees networking, only the story.
- **Agent-side question:** if we give a tiny model a simplified, everyday abstraction of a more complex system, can it operate that system correctly through the abstraction? Does the abstraction make learning easier than raw network text?
- **GPU-side question:** *where should the agent's memory live?* I build the agent loop in 4 versions (V0 to V3), each moving more of it onto the GPU and into faster memory. I profile each version and compare them.
- **The two questions connect:** a shorter, simpler abstraction means fewer tokens per episode, which means a smaller memory (KV cache) per agent on the GPU.
- **Hardware:** RunPod GPU for training and the main experiments. Jetson Nano 2GB as the small edge device, where the memory limit becomes an experiment of its own.
- **Next deadline:** quick project review on **Nov 25–26**. Bring a 3-minute pitch and 1–2 slides.

---

## 1. What the course actually wants

### Assessment (intro slides, p.15–21)

| Part | Weight | What it is |
|---|---|---|
| Theory exam | 30% | PC-based written exam on parallel programming and GPU architecture |
| Project assignment | 40% | max 3 students per group |
| Presentation | 30% | written report + slides + speech (communication skills count) |

### What the project should do (slide p.18)

- Target: **apply data-parallelism and CUDA concepts**.
- **Evaluate and analyze metrics (performance)**, and improve existing GPU code or architectures.
- Accepted project types:
  - implement a GPU version of some computationally intensive CPU code
  - **reproduce some existing GPU research work** ← KV caching / attention kernels fit here
  - **do novel GPU research** ← the agent-memory study fits here

### Important dates

| Date | What |
|---|---|
| Nov 4, 2026 | Project assignment presentation |
| **Nov 25–26, 2026** | **Quick review of the project (3-min description, 1–2 slides)** |
| Dec 16–17, 2026 | Assignment verification |
| Jan 7–8, 2027 | Assignment final check-up |
| Jan 11 – Feb 20, 2027 | Exam session 1 (others: Jun 7–Jul 17, Aug 31–Sep 18) |

- Every group **must get feedback in at least one verification session**.
- Book the exam and deliver the project by the exam date. The presentation and oral happen the week after.
- Example proposals for reference: `course/example-proposals/`.

---

## 2. Decisions made so far

| Decision | Choice | Why |
|---|---|---|
| Framework | **PyTorch for training only (on the Mac). Everything that runs the agent is my own CUDA C.** | The new part is the agent memory, not training; the graded GPU work is the engine. PyTorch also becomes the correctness reference |
| Environment | **A delivery-agent story world** (doors, roads, bus stations, houses) running on top of a **deterministic network state machine** | Everyday words for the agent; real network concepts underneath |
| Networking | **Purely simulated.** No real interfaces or routers. | Scope. Real networking is future work |
| CPU baseline | **No speed comparison against the CPU.** V0 (naive GPU) is the baseline. | We have the profiling lecture; compare GPU versions against each other |
| CPU code | A small C version is still useful, **only for checking correctness** (are my CUDA numbers right?) | Hand-written kernels are easy to get subtly wrong |
| Profiling | Central. It's how every version gets measured. | Lecture 7 + Lab 5 |
| Visualization | Possibly, via CUDA–OpenGL interop. **Stretch goal.** | Great for the presentation (30%), but extra scope |

### Important nuance: profiling ≠ "the agent's mind"

- **Profilers** (Nsight Systems, Nsight Compute, nvprof) measure **how the GPU runs**: time per kernel, memory bandwidth, occupancy, stalls. Example: "the attention kernel takes 40% of each decision."
- **"What the agent is thinking"** means its internal numbers, such as attention weights (which earlier words it focused on) and activations. To see those, copy them out of GPU memory and **visualize** them.
- So: profiling answers "is it fast and why?" Visualization answers "what is it doing?" Both can go in the presentation.

---

## 3. The project idea in detail

### 3.1 Two layers: the story world on top, the network underneath

```
           what the agent sees                     what really happens
  ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
  │  STORY WORLD (text)              │    │  NETWORK STATE MACHINE           │
  │  doors, roads, bus stations,     │◄──►│  interfaces, links, routers,     │
  │  bus lines, streets, houses      │    │  routes, subnets, hosts          │
  └──────────────────────────────────┘    └──────────────────────────────────┘
                     ▲         translator (deterministic, both directions)
                     │
                  the agent
```

- **The network state machine is the ground truth.** It holds the real state: which interfaces are UP, which routes exist, which host is the destination.
- **The translator** has two jobs:
  1. **state → story:** describes the current network state as a scene ("door A is closed", "the board shows no line to Maple Street").
  2. **story action → network operation:** "open door A" becomes `eth0 → UP`.
- **The agent only ever sees the story.** It doesn't know it is configuring a network.

### 3.2 The mapping

| Story world | Network equivalent | Example action → operation |
|---|---|---|
| Depot (where the courier starts) | The source machine | — |
| Door of the depot (A, B, …) | Network interface (eth0, eth1, …) | `open door A` → `ip link set eth0 up` |
| Closed / open door | Interface DOWN / UP | `close door A` → `ip link set eth0 down` |
| Road from a door | Network link / cable | `walk road A` → use the link on eth0 |
| Bus station | Router | — |
| Timetable board at the station | Routing table | `read board` → `ip route show` |
| Bus line to a street | A route to a subnet | `ask for line to maple` → `ip route add 10.0.2.0/24 via …` |
| Taking a bus | Forwarding the packet along that route | `take line 7` → packet goes via the router |
| Street / district | Subnet (10.0.2.0/24) | — |
| House number | Host address (10.0.2.5) | — |
| Knocking on the house / delivering the parcel | Ping / connection established | `knock house 5` → `ping 10.0.2.5` succeeds |

Possible extras for harder scenarios (stretch):

| Story world | Network equivalent |
|---|---|
| Locked garden gate | Firewall rule blocking traffic |
| Getting a key from the gatekeeper | Adding an allow rule |
| Address book ("who lives at Maple 5?") | DNS lookup |
| Road closed for repairs | Link down / broken cable |
| Wrong address on the parcel | Misconfigured destination |

### 3.3 Example episode (the text the model actually sees)

```
<s> you are at the depot . parcel for house 5 on maple street .
SEE door A closed | door B closed
OPTIONS 1 open door A 2 open door B 3 wait
ACT 1
SEE door A open . road A leads to north station
OPTIONS 1 walk road A 2 close door A 3 open door B
ACT 1
SEE north station . board shows line 1 to oak street . no line to maple street
OPTIONS 1 take line 1 2 ask for line to maple 3 go back
ACT 2
SEE line 7 to maple street added
OPTIONS 1 take line 7 2 take line 1 3 go back
ACT 1
SEE maple street . house 5
OPTIONS 1 knock house 5 2 go back
ACT 1
RESULT parcel delivered DONE
```

What the network state machine did behind the scenes, step by step:

| Agent action | Network operation | Network state after |
|---|---|---|
| open door A | `ip link set eth0 up` | eth0 UP |
| walk road A | reach the router on eth0's link | next hop = r1 |
| ask for line to maple | `ip route add 10.0.2.0/24 via r1` | route to 10.0.2.0/24 exists |
| take line 7 | forward along the new route | packet in subnet 10.0.2.0/24 |
| knock house 5 | `ping 10.0.2.5` | reachable → **goal reached** |

### 3.4 Honest note: does "regular text" make the model understand better?

- **Not by itself, for a model trained from scratch.** Our model has never read any human text before training. To it, "door" is just token #12, exactly as meaningless as "eth0". Everyday words would only bring built-in understanding with a *pretrained* model (future work).
- **What the abstraction can genuinely help with:**
  1. **Hiding irrelevant detail.** Numbers like `10.0.2.0/24 via 10.0.1.1` become "line to maple street". There are fewer distinct tokens to learn and fewer tokens per step.
  2. **A small, consistent set of concepts and actions** ("open", "walk", "take", "knock"), reused everywhere.
  3. **Shorter episodes.** That means a smaller KV cache, so less GPU memory per agent. This is where the agent side meets the GPU side.
  4. **Readability** for us when debugging, and for the presentation.
- **Whether the story version actually learns faster or generalizes better than raw network text is an experiment, not an assumption** (see 3.5). "No difference" would still be a valid, honest result.

### 3.5 The agent-side research questions

| # | Question | How to test it |
|---|---|---|
| A1 | Can a tiny model trained **only on the story world** correctly operate the network through the translator? | Run the agent; check the **network** state at the end (is the destination reachable?), not just the story |
| A2 | Does the story abstraction help compared with raw network text? | Train **two agents with the same model size**: one on story text, one on raw network text (`eth0 DOWN`, `route add 10.0.2.0/24 …`). Compare success rate on unseen scenarios, training steps to learn, **tokens per episode** |
| A3 | Does it generalize instead of memorizing? | Test on **unseen scenarios**: more doors, different bus networks, longer routes, new street/house combinations |
| A4 (future) | Can the same agent drive a different system by swapping only the translator? | E.g. reuse the delivery world for a different network topology or a different domain |

Note on A1: the translator is exact, so "solved the story" always means "solved the network". The real test is **A3**: does it still work when the scenarios get bigger or different?

### 3.6 The simulator on the GPU

The network state machine is a lookup table, which fits the GPU nicely:

```c
struct Transition { int next_state; int text_id; };   // text_id → story tokens to print

__constant__ int        n_options[MAX_STATES];               // how many options per state
__constant__ Transition table[MAX_STATES][MAX_OPTIONS];      // state × option → next
```

- **Constant memory:** the transition table fits there.
- **Built-in experiment:** constant memory is fast when all threads in a warp read the **same** address (broadcast) and slow when they read **different** addresses (reads get serialized). With many agents in different states, that's a real case to measure: constant vs shared vs read-only cache (`__ldg`) vs global.
- **Divergence:** agents in different states take different code paths inside the simulator kernel.
- The translator's story text can be **pre-generated** (each state has a fixed token list), so the GPU only copies token IDs. No string handling inside kernels.

Ways to make it harder: more doors and stations, multi-hop routes (change buses), dependencies (open the door before walking the road), dead ends, the extras from 3.2.

### 3.7 The agent never writes free text

- The model doesn't need to generate sentences. At `ACT` it only scores the **allowed option numbers** and picks the best one.
- This is called masking the logits: ignore every score except the allowed options.
- Result: one forward pass per decision, and the agent can never pick an invalid action.

### 3.8 Custom tokenizer

- The simulator produces all the text, so the vocabulary is **tiny and known in advance** (about 100–300 words: `door`, `open`, `maple`, `5`, `ACT`, …).
- The tokenizer is just a lookup table from word to ID. No BPE or anything fancy.

### 3.9 The GPU-side research question: agent memory

At every step, the conversation grows (new scene + options + action). The model has to remember the whole episode. In a transformer, that memory is the **KV cache**: the saved K and V vectors of all previous tokens, so they don't get recomputed.

**Where that memory lives, and where the simulator runs, is the experiment:**

| Version | What changes | What it shows |
|---|---|---|
| **V0, naive** | CPU runs the simulator; GPU recomputes the **whole** conversation every step; scores copied back to the CPU every step | Baseline. How much time goes to transfers and wasted recomputation |
| **V1, KV cache in global memory** | Only new tokens get processed; old K/V are reused | Savings from caching |
| **V2, whole loop on the GPU** | The simulator step runs inside a kernel; nothing goes back to the CPU between steps (maybe CUDA Graphs) | Cost of CPU↔GPU round-trips |
| **V3, KV cache in shared memory** | Persistent kernel, **one block per agent**; the KV cache sits in shared memory next to the cores | Latency vs occupancy trade-off |

Optional extras: FP16 KV cache (half the memory), a sliding-window memory (keep only the last N tokens), and comparing a short state summary against the full transcript.

#### Rough memory math (to sanity-check the experiments)

Model: d = 128, 4 layers, context 512 tokens.

- KV cache per agent = 2 (K and V) × 4 layers × 512 tokens × 128 × bytes per value
  - FP32: **2 MB per agent**
  - FP16: **1 MB per agent**
- Episode length: a story step is roughly 25–35 tokens, so a 10-step delivery is ~300 tokens. Raw network text (IP addresses, masks) would likely be longer. That's experiment A2.
- **Jetson Nano 2GB:** the OS and desktop take a large share of the 2 GB, which is shared between CPU and GPU. Realistically only a few hundred agents fit at FP32. That makes "how many agents fit, and which layout fits the most?" a real, measurable question.
- **Shared memory (V3):** one layer's K and V for a 128-token context in FP16 = 2 × 128 × 128 × 2 B = **64 KB**. That fits in one block on modern GPUs (up to ~100–228 KB per SM, depending on the GPU), but it lowers occupancy. That trade-off is the V3 experiment.

### 3.10 Visualization (stretch goal)

- A CUDA kernel draws directly into an OpenGL buffer (CUDA–OpenGL interop).
- Ideas:
  - couriers walking through the town map (doors, roads, stations, houses) live, with the matching network state shown next to it
  - an attention heatmap: which earlier words the agent looked at when it chose an action (the "agent's mind" view)
- Works on the **Jetson**, which has a display output. **RunPod has no screen**, so render to image files or a video there.

---

## 4. LLM primer from zero (how all of this actually works)

### 4.1 The model is a function full of numbers

```
input:  a list of word IDs      [5, 17, 3, 42, ...]
output: a score for every word in the vocabulary, meaning "how likely is this word next?"
```

**The weights are just arrays of floats.** That's all a trained model is. For our tiny model (d = 128 numbers per word, 4 layers, 256-word vocabulary):

```
token embedding     [256 × 128]    one row of 128 numbers per word
position embedding  [512 × 128]    one row per position (0..511)
per layer (×4):
  Wq, Wk, Wv, Wo    [128 × 128]    attention matrices
  W1                [128 × 512]    MLP, expands
  W2                [512 × 128]    MLP, shrinks back
  ln gains/biases   [128]          normalization
unembedding         [128 × 256]    vector back to word scores (often shares weights with the token embedding)
```

Parameter count:
- per layer: 4 × 128² + 2 × 128 × 512 ≈ **197K**
- × 4 layers ≈ 790K
- plus embeddings ≈ 100K
- **total ≈ 0.9M floats ≈ 3.6 MB at FP32**

- **Saving the model** = writing those arrays to a binary file (`agent.bin`).
- **Loading the model** = reading the file and `cudaMemcpy`-ing it into GPU memory.

### 4.2 Text → numbers (tokens)

Every word the story world can print gets an ID from my lookup table:

```
SEE door A closed | door B closed OPTIONS 1 open door A 2 open door B 3 wait ACT
→ [7, 12, 30, 15, 4, 12, 31, 15, 2, 20, 40, 12, 30, 21, 40, 12, 31, 22, 41, 3]
```

### 4.3 The forward pass: weights + input → output

Following those 20 token IDs through the model:

```
1. Embedding lookup: each ID picks its row, plus its position row
                                                → 20 vectors of 128 floats    [20 × 128]
2. Repeat for each of the 4 layers:
   a. LayerNorm:   rescale each vector                                         (reduction kernel)
   b. Attention:   Q = x·Wq,  K = x·Wk,  V = x·Wv                              (3 matmuls)
                   score[i][j] = Q[i]·K[j]   "how relevant is word j to word i?"
                   softmax over scores, only for words BEFORE i (causal mask)  (reduction kernel)
                   out = weighted mix of the V vectors, then ·Wo               (matmuls)
                   → this is how "ACT" can look back and notice "door A closed"
   c. LayerNorm again
   d. MLP:         x·W1 → GELU → ·W2                                           (2 matmuls)
   (each sub-block adds its result back onto x: the "residual connection")
3. Take the LAST vector (the one at "ACT"), multiply by the unembedding → 256 scores
4. Mask: keep only allowed options, e.g. "1" = 4.2, "2" = -1.0, "3" = 0.3 → pick "1"
```

**It's almost all matrix multiplies, softmax and layernorm.** Those are tiled matmul and reductions with warp shuffles, straight from the labs. That's why this fits a GPU course.

Intuition for attention: every token sends out a question (Q), every token carries a label (K) and some content (V). A question matches against all the labels, and the best matches contribute the most content. The KV cache simply stores the labels and content of old tokens so they don't have to be recomputed.

### 4.4 Training: where the weight values come from

At first the weights are random numbers, so the model picks options at random.

**Step A: generate data (imitation learning, no reinforcement learning needed).**
- The network state machine is deterministic, so a simple script (an "expert") can always compute the correct action, for example by BFS on the state graph toward "destination reachable".
- Run thousands of random scenarios (different doors, stations, streets, houses) with the expert, and record full episodes as story text.
- The model learns to copy the expert. This is called **behavior cloning**, and it's much easier than RL.

**Step B: the training loop, repeated many thousands of times:**

```
1. forward pass on a batch of episodes
2. loss = how surprised the model was by each correct next token
        = -log(probability it gave the right token), averaged
   (for an agent: mostly care about the tokens right after "ACT")
3. backward pass: for EVERY weight, compute "if I nudge this up,
   does the loss go up or down, and by how much?" (= the gradient)
   → the chain rule; PyTorch computes this automatically (loss.backward())
4. update: weight -= learning_rate × gradient     (in practice: the AdamW optimizer)
```

- The loss goes down, and eventually the model reliably puts the highest score on the expert's action.
- With ~1M parameters: **roughly 10–30 minutes on my M1 Mac** (PyTorch, `device="mps"`).
- Hold out scenarios the model never saw during training to check it learned the rules instead of memorizing episodes (question A3).

### 4.5 After training: running the agent (inference)

The training code is no longer needed. Load the weights and run **only the forward pass**, in a loop with the simulator:

```c
load_weights("agent.bin");                 // cudaMemcpy ~4 MB onto the GPU
text   = sim_reset(scenario);              // "you are at the depot ... OPTIONS ... ACT"
tokens = tokenize(text);

while (!done) {
    scores = forward(tokens);              // all the CUDA kernels from 4.3
    action = best_allowed_option(scores);  // e.g. "1" = open door A
    text   = sim_step(action);             // translator: eth0 → UP, then describes the new scene
    tokens += tokenize(action + text);     // the conversation (the agent's memory) grows
}
check_network_state();                     // question A1: is 10.0.2.5 actually reachable?
```

- **V0** re-runs `forward` over all of `tokens` every iteration, and `sim_step` runs on the CPU.
- **V1–V3** turn this loop into the agent-memory experiment from section 3.9.

### 4.6 The CUDA kernels I'd write, and which course topic each one uses

Training happens in PyTorch, so these are all **forward-pass (inference)** kernels for the agent engine.

| Kernel | Used in | Course topic |
|---|---|---|
| Tiled matrix multiply | every layer | shared memory, tiling, coalescing (lecture 5, Lab 4) |
| Softmax | attention, final scores | reductions, warp shuffle (lectures 6, 9) |
| LayerNorm | every layer | reductions, cooperative groups (lecture 9) |
| Attention (with KV cache) | every layer | memory hierarchy (lecture 5) |
| Embedding lookup | input layer | gather (indexed reads) |
| GELU, residual add | every layer | elementwise, memory-bound (roofline, lecture 7) |
| Simulator step | agent loop | **constant memory**, divergence (lectures 4, 5) |
| Counting actions / delivered parcels across agents | evaluation | **atomics**, histogram (lectures 6, 8) |
| Running agent groups concurrently | multi-agent runs | **streams**, async copies (lecture 8, Lab 6) |
| Comparison against cuBLAS | evaluation | libraries (week 11) |

---

## 5. WIP Proposal

**Working title:**
**"Where Should an Agent Remember? On-GPU Memory for Tiny Transformer Agents in a Custom CUDA Engine"**

**Problem statement (draft):**
Small language models can act as decision-making agents: at every step they read an observation and choose an action. This project trains a tiny transformer agent and runs it with our own CUDA inference engine, with no deep-learning framework at run time. The agent is a courier in a simple text world (doors, roads, bus stations, houses), and that world is an abstraction of a deterministic network simulator underneath: opening a door brings an interface up, a new bus line is a new route. The agent loop is a poor match for the GPU. Each decision is small and sequential, the context grows every step, and naive implementations recompute the whole history and bounce data between CPU and GPU. We study how the placement of the agent's memory (the KV cache) and of its environment affects latency, throughput and memory footprint, on a datacenter GPU and on an embedded Jetson Nano. We also study whether the simplified abstraction lets a tiny model operate the underlying system correctly, using fewer tokens than raw network text.

**Implementation:**
- Model trained in PyTorch; weights exported to a plain binary file.
- Custom CUDA inference engine: tiled GEMM, softmax/layernorm with warp-level reductions, attention over a KV cache, all checked against PyTorch outputs.
- A deterministic network state machine with a story-world translator, running on the GPU with its transition table in constant memory.
- Four agent-loop versions (V0–V3): naive recomputation → KV cache in global memory → fully on-GPU loop → KV cache in shared memory with persistent kernels.

**KPIs:**
- Decisions per second; latency per decision
- Speedup of each version over V0, and vs PyTorch running the same model
- Bytes of memory per agent; maximum number of concurrent agents (Jetson 2GB vs RunPod GPU)
- Kernel efficiency: percentage of cuBLAS speed, position on the roofline, occupancy, achieved bandwidth (Nsight Compute / nvprof)
- Timeline analysis of transfers vs compute (Nsight Systems)
- Agent side: success rate on unseen scenarios (checked on the real network state); story text vs raw network text (success rate, tokens per episode)

**Hardware:**
- RunPod GPU (exact model TBD)
- Jetson Nano 2GB (Maxwell, SM 5.3, 2 GB shared memory, JetPack 4.6 / CUDA 10.2)

**Stretch goal:** live CUDA–OpenGL visualization of couriers moving through the town, and of attention maps.

**Future work:** connect the translator to real network operations; reuse the agent for other systems by swapping the translator.

**References:**
- Vaswani et al. (2017), *Attention Is All You Need*
- Karpathy, *llm.c*: https://github.com/karpathy/llm.c
- Dao et al. (2022), *FlashAttention*

**Possible group split (if 2–3 people):**
- Person 1: training in PyTorch + GEMM and attention kernels
- Person 2: story world + network state machine on the GPU, the agent loop V0–V3
- Person 3: Jetson port, FP16, profiling, visualization

---

## 6. Hardware notes and risks

### RunPod
- Good for training and for the main experiments.
- **Risk:** rented containers often **block GPU performance counters**, so Nsight Compute (`ncu`) fails with `ERR_NVGPUCTRPERM`. You can't change the host driver setting from inside a container. (One of the example proposals, Angry Santa, hit this exact problem.)
- **Do this first:** rent the cheapest pod for an hour and run `ncu` on a toy kernel. If it fails, look for a provider or pod type that allows counters, or rely on Nsight Systems + nvprof-style timing.
- Nsight Systems (the timeline) usually works without counter access.

### Jetson Nano 2GB
- Maxwell GPU, **SM 5.3**, 128 CUDA cores, **2 GB of memory shared between CPU and GPU**.
- Stuck on **JetPack 4.6 / CUDA 10.2 / Ubuntu 18.04**. Modern PyTorch doesn't run on it, which is another reason plain CUDA C is the right call.
- **Nsight Compute doesn't support Maxwell.** Profile with `nvprof` there (Lab 5 covers it).
- Fine for **inference** and edge measurements (latency, memory per agent). **Bad for training.** Train on RunPod and copy `agent.bin` over.
- Supports FP16, so it's worth testing an FP16 KV cache and FP16 weights there.
- Watch out for CUDA 10.2 limits when writing code that has to build on both machines: avoid newer CUDA features, or put them behind `#if`.

### Scope risks
- **Biggest risk:** learning how LLMs work from zero before Dec 16.
  - **Mitigation:** training is in PyTorch (no hand-written backward passes); all custom effort goes into the CUDA engine, the agent-memory versions and the abstraction study.
- **Correctness:** hand-written kernels go subtly wrong easily. Dump PyTorch's intermediate outputs for a fixed input and compare every kernel's output against them.
- **Keep the model tiny.** ~1M parameters is enough for this task. Making it bigger doesn't make the GPU story better.
- **Experiment A2 doubles the training runs** (story vs raw). Cheap at this model size, but plan for it.
- **Visualization is optional.** Do it only after V0–V3 work.

---

## 7. Learning path (I start from zero on LLMs)

1. **Karpathy, "Let's build GPT: from scratch, in code, spelled out"** on YouTube (~2 hours). Best zero-to-understanding resource. It uses PyTorch, which is exactly what we use for training.
2. Optional: Karpathy's **"Zero to Hero"** series, especially the backprop videos ("micrograd", "becoming a backprop ninja"), to understand what `loss.backward()` actually does.
3. **Read llm.c's `train_gpt2.c`** (~1,000 lines of plain C, no libraries). Its forward pass is exactly what my CUDA engine reimplements.
4. Then llm.c's CUDA version (`train_gpt2.cu` and its `dev/cuda/` kernels): see how each CPU function becomes kernels.
5. Re-check course material while building: lecture 5 (memory hierarchy), 6 (atomics, reductions), 7 (profiling, roofline), 8 (streams), 9 (cooperative groups / warp shuffle), Labs 4–6.

---

## 8. Rough timeline

| When | Goal |
|---|---|
| Nov 19–24 | Watch the Karpathy video; test `ncu` on RunPod; finalize the proposal PDF + 1–2 slides |
| **Nov 25–26** | **Quick review: pitch it, get the professor's OK or changes** |
| Late Nov – early Dec | First LLM on the Mac (tiny Shakespeare); then network state machine + story translator + expert + dataset; tokenizer; train the courier agent in PyTorch |
| Early–mid Dec | Export `agent.bin`; CUDA engine V0 + V1 on RunPod; first profiling numbers |
| **Dec 16–17** | **Verification: show a trained courier delivering (and the network state it produced), V0 vs V1 numbers** |
| Late Dec – early Jan | V2, V3; Jetson port; story vs raw-text experiment; full profiling |
| **Jan 7–8** | **Final check-up** |
| Before exam date | Report, slides, (visualization if time allows) |

---

## 9. Open questions

- [ ] Solo or group? Who? (Max 3. A group makes V0–V3 + Jetson + abstraction study + visualization much more realistic.)
- [ ] Visualization: stretch goal or cut?
- [ ] Is the Jetson Nano 2GB set up and working (JetPack flashed, `nvcc` + `nvprof` run)?
- [ ] Which RunPod GPU, and does `ncu` work on it?
- [ ] Is it OK with the professor that training uses PyTorch and the graded GPU work is the custom inference engine + agent-memory study? Ask at the Nov 25–26 review.
- [ ] Final story-world design: how many doors/stations/streets, which extras (gates = firewall, address book = DNS), what makes a scenario "unseen"?
- [ ] Raw network text format for experiment A2 (needs to describe exactly the same states as the story version).
