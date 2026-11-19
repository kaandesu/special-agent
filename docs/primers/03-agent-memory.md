# Primer 3 — What "agent memory on the GPU" actually means

## 1. The model itself has no memory

After training, the model is a fixed function: **tokens in → scores out**. It remembers nothing between calls. The only reason it "knows" door A is already open is that you **feed it the whole story so far, every time**.

So the agent's "memory" is the episode history: every scene, option and action since the start.

## 2. The naive way recomputes everything

Each step adds about 30 tokens (new scene + options + action):

```
step 1:  forward pass over   30 tokens  → choose
step 2:  forward pass over   60 tokens  → choose   (the first 30 again!)
step 3:  forward pass over   90 tokens  → choose   (the first 60 again!)
...
step 10: forward pass over  300 tokens  → choose
```

In total: **1,650 tokens processed to make 10 decisions**, and only 300 of them were new. The longer the episode, the worse it gets. Every step also copies the scores back to the CPU, runs the simulator there, and copies the new tokens back to the GPU.

That's **V0**.

## 3. The trick: K and V of old tokens never change

Inside attention, every token gets a **K** (its label) and a **V** (its content). A token can only look at words *before* it, so once "door A closed" has been processed, its K and V are **final forever**. New tokens never change them.

So you save them. That's the **KV cache**, literally two big arrays on the GPU:

```c
float kcache[N_LAYERS][MAX_TOKENS][128];   // K of every token seen so far
float vcache[N_LAYERS][MAX_TOKENS][128];   // V of every token seen so far
int   pos;                                  // how many tokens are stored
```

Each step then only processes the **new** tokens:

```c
// one decision, V1
for (l = 0; l < N_LAYERS; l++) {
    q, k, v = x_new · Wq, Wk, Wv;                 // only the ~30 NEW tokens
    kcache[l][pos .. pos+30] = k;                  // append to memory
    vcache[l][pos .. pos+30] = v;
    scores = q · kcache[l][0 .. pos+30]ᵀ;          // new tokens look at ALL history
    out    = softmax(scores) · vcache[l][0 .. pos+30];
    ...
}
pos += 30;
```

10 decisions now process 300 tokens instead of 1,650. **The KV cache *is* the agent's memory, made concrete as an array.**

It costs memory: 2 (K and V) × 4 layers × 512 tokens × 128 × 4 bytes = **2 MB per agent at FP32** (1 MB at FP16).

That's **V1**.

## 4. The GPU question: *where* does that array live?

The attention kernel reads the **whole** cache at every step, so where it sits decides how fast every decision is. This is lecture 5's memory hierarchy:

| Memory | Size | Speed | Who can use it |
|---|---|---|---|
| Registers | tiny, per thread | fastest | one thread |
| **Shared memory** | ~48–228 KB per SM | very fast | one block |
| L2 cache | a few MB | fast | whole GPU |
| **Global memory** | GBs | slow (hundreds of cycles) | whole GPU |
| **CPU RAM, over PCIe** | lots | slowest | needs a `cudaMemcpy` |

The versions walk down this table:

- **V0:** no cache; the simulator lives on the CPU, so there's PCIe traffic every step.
- **V1:** cache in **global memory**, no recomputation. The simulator is still on the CPU.
- **V2:** the **simulator moves onto the GPU too**. The game state is a few integers per agent; the transition table lives in **constant memory**. The whole loop (decide → step → decide → step) runs without the CPU.
- **V3:** cache in **shared memory**. One block runs one agent for the whole episode (a "persistent kernel"), with its memory right next to the cores.

## 5. V3 is where the real research trade-off is

Shared memory is small:
- **Jetson** (Maxwell): 48 KB per block. One layer's K+V at FP16 for 128 tokens = 2 × 128 × 128 × 2 B = 64 KB, so **it doesn't even fit**.
- **A big RunPod GPU** (~100–228 KB): it fits, but each agent takes so much shared memory that fewer agents run at once (lower **occupancy**).

So there are design choices, and each is something to measure:
- keep only the most recent tokens in shared memory, older ones in global
- FP16 to halve the size
- keep one layer at a time in shared memory

## 6. What you'd actually see

**Nsight Systems timeline, V0:**
```
GPU:  [forward]          [forward]          [forward]
PCIe:          [copy][copy]       [copy][copy]
CPU:               [sim]              [sim]          ← GPU sits idle here
```
**V2:** `[forward][sim][forward][sim][forward]...` packed tightly, no gaps.

## 7. The graphs this produces
1. **Time per decision vs step number.** V0 keeps rising as the episode grows; V1–V3 stay almost flat.
2. **Decisions per second** V0 → V3, with 1 agent and with 1,000 agents in parallel.
3. **Memory per agent** and **how many agents fit**, Jetson vs RunPod.
4. **Nsight Compute for the attention kernel**, V1 vs V3: bandwidth, occupancy, roofline.
5. **Story vs raw network text:** fewer tokens per episode → smaller cache → more agents fit. This is where the agent-side question meets the GPU-side question.

## In one sentence
Take the weights from the Mac, write your own CUDA program that runs the courier agent, then move its memory and its simulator closer and closer to the GPU cores, profiling each step.
