# data/ — training data (Phases 1 and 3)

- `gen_data.py` (Phase 3): runs the expert on random scenarios and writes episodes.
- All outputs go to `data/generated/`, which is **git-ignored**. Regenerate instead of committing.

Planned outputs:
- `shakespeare.txt` (Phase 1)
- `story_train.txt`, `story_test_seen.txt`, `story_test_unseen.txt`
- `raw_train.txt`, `raw_test_seen.txt`, `raw_test_unseen.txt` (same scenarios, raw network text)
- `vocab.json`

"Unseen" means scenario **types** held out from training (bigger towns, more hops), not just new random seeds.
