# train/ — PyTorch, runs on the Mac (Phases 1, 2, 4)

Nothing here yet. Setup: see Phase 0 in [`../docs/PLAN.md`](../docs/PLAN.md) (Python 3.12 venv, `device="mps"`).

Planned files:
- `data_shakespeare.py`: tiny Shakespeare, character vocabulary, train/val split (Phase 1)
- `model.py`: tiny GPT (d=128, 4 layers, 4 heads), nanoGPT-style (Phase 1, reused in Phase 4)
- `train.py`: training loop for Shakespeare (Phase 1)
- `chat.py`: type a prompt, watch it continue (Phase 2)
- `tokenizer.py`: word-level vocabulary for the courier world (Phase 3)
- `train_agent.py`: training on courier episodes (Phase 4)
- `play.py`: agent loop in Python, masked options, success rate checked on the network state (Phase 4)
- `export.py`: write `checkpoints/agent.bin` (Phase 4)
- `reference.py`: dump every intermediate output for a fixed input, for checking the CUDA kernels (Phase 4)

## agent.bin format (to fill in when export.py exists)
```
header:  magic "SAGT" | version | vocab_size | d_model | n_layers | n_heads | ctx_len   (int32 each)
tensors: float32, little-endian, row-major, fixed order:
         tok_emb, pos_emb,
         per layer: ln1_w, ln1_b, Wq, Wk, Wv, Wo, ln2_w, ln2_b, W1, b1, W2, b2,
         lnf_w, lnf_b   (unembedding tied to tok_emb)
```
`cuda/weights.cu` must read exactly this order.
