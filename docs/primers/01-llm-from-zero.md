# Primer 1 — How an LLM works, from zero

Using the courier world as the example throughout.

## 1. The model is a function full of numbers

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
unembedding         [128 × 256]    vector back to word scores (usually shares weights with the token embedding)
```

Parameter count:
- per layer: 4 × 128² + 2 × 128 × 512 ≈ **197K**
- × 4 layers ≈ 790K
- plus embeddings ≈ 100K
- **total ≈ 0.9M floats ≈ 3.6 MB at FP32**

- **Saving the model** = writing those arrays to a binary file (`agent.bin`).
- **Loading the model** = reading the file into memory (on the GPU: `cudaMemcpy`).

## 2. Text → numbers (tokens)

Every word the simulator can print gets an ID from a lookup table that I write:

```
SEE door A closed | door B closed OPTIONS 1 open door A 2 open door B 3 wait ACT
→ [7, 12, 30, 15, 4, 12, 31, 15, 2, 20, 40, 12, 30, 21, 40, 12, 31, 22, 41, 3]
```

## 3. The forward pass: weights + input → output

Following those 20 token IDs through the model:

```
1. Embedding lookup: each ID picks its row, plus its position row
                                                → 20 vectors of 128 floats    [20 × 128]
2. Repeat for each of the 4 layers:
   a. LayerNorm:   rescale each vector
   b. Attention:   Q = x·Wq,  K = x·Wk,  V = x·Wv
                   score[i][j] = Q[i]·K[j]   "how relevant is word j to word i?"
                   softmax over scores, only for words BEFORE i (causal mask)
                   out = weighted mix of the V vectors, then ·Wo
                   → this is how "ACT" can look back and notice "door A closed"
   c. LayerNorm again
   d. MLP:         x·W1 → GELU → ·W2
   (each sub-block adds its result back onto x: the "residual connection")
3. Take the LAST vector (the one at "ACT"), multiply by the unembedding → 256 scores
4. Mask: keep only allowed options, e.g. "1" = 4.2, "2" = -1.0, "3" = 0.3 → pick "1"
```

**It's almost all matrix multiplies, softmax and layernorm.** Those are exactly the kernels from the GPU labs (tiled matmul, reductions with warp shuffles).

**Intuition for attention:** every token sends out a question (Q), and every token carries a label (K) and some content (V). A question matches against all the labels, and the best matches contribute the most content.

## 4. Training: where the weight values come from

At first the weights are random numbers, so the model picks options at random.

**Step A: generate data (imitation learning, no reinforcement learning needed).**
- The simulator is deterministic, so a simple script (the "expert") always knows the correct action: BFS on the state graph toward "destination reachable".
- Run thousands of random scenarios with the expert and record full episodes as text.
- The model learns to copy the expert. This is called **behavior cloning**.

**Step B: the training loop, repeated many thousands of times:**

```
1. forward pass on a batch of episodes
2. loss = how surprised the model was by each correct next token
        = -log(probability it gave the right token), averaged
3. backward pass: for EVERY weight, compute "if I nudge this up,
   does the loss go up or down, and by how much?" (the gradient)
   → PyTorch does this automatically: loss.backward()
4. update: weight -= learning_rate × gradient   (in practice: the AdamW optimizer)
```

- The loss goes down, and eventually the model reliably puts the highest score on the expert's action.
- With ~1M parameters: roughly 10–30 minutes on the M1 Mac.
- Hold out scenario types the model never saw, to check it learned the rules instead of memorizing episodes.

## 5. After training: running the agent (inference)

Training code is no longer needed. Load the weights and run **only the forward pass**, in a loop with the simulator:

```c
load_weights("agent.bin");
text   = sim_reset(scenario);              // "you are at the depot ... OPTIONS ... ACT"
tokens = tokenize(text);

while (!done) {
    scores = forward(tokens);              // section 3
    action = best_allowed_option(scores);  // e.g. "1" = open door A
    text   = sim_step(action);             // translator: eth0 → UP, then describes the new scene
    tokens += tokenize(action + text);     // the conversation (the agent's memory) grows
}
check_network_state();                     // is 10.0.2.5 actually reachable?
```

## 6. Why the Shakespeare model "continues" instead of chatting

A model only learns the shape of its training text. Trained on Shakespeare, it continues text in Shakespeare's style. ChatGPT-style chatting comes from training on chat-shaped text. The courier agent is the same idea: it's trained on episode-shaped text, so at `ACT` it "continues" with an action.

## Learning resources
1. Karpathy, **"Let's build GPT: from scratch, in code, spelled out"** (YouTube, ~2 h)
2. Karpathy, **nanoGPT** (`model.py`, ~300 lines): https://github.com/karpathy/nanoGPT
3. Karpathy, **"Zero to Hero"** series: the micrograd and backprop videos explain what `loss.backward()` does
4. Karpathy, **llm.c** `train_gpt2.c`: a GPT forward pass in plain C, close to what the CUDA engine reimplements: https://github.com/karpathy/llm.c
