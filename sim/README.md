# sim/ — the courier world (Phase 3)

Python. Nothing here yet.

Planned files:
- `network.py`: network state machine, the ground truth (interfaces UP/DOWN, links, routers, routes, destination host; `is_goal()`)
- `story.py`: translator, network state ↔ story text (doors, roads, bus stations, bus lines, streets, houses)
- `raw.py`: translator, network state → raw network text (`eth0 DOWN`, `route add …`), for experiment A2
- `expert.py`: BFS over states → shortest correct action sequence
- `scenarios.py`: random scenario generator with difficulty knobs; held-out "unseen" scenario types
- `play_human.py`: play the world by hand in the terminal

Design and mapping table: [`../docs/project-notes.md`](../docs/project-notes.md) §3. Steps: [`../docs/PLAN.md`](../docs/PLAN.md) Phase 3.

Keep the transition table a flat integer table `next_state[state][option]`: it becomes `__constant__` memory in `cuda/`.
