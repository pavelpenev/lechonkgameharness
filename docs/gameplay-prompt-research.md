# Gameplay prompt profile research

The `gameplay_v1` profile uses live candidate legality and current snapshots as
the contract, while treating strategy advice as conditional. This matters
because the harness may expose a modified or ruleset-dependent Freeciv surface;
generic rules knowledge must never override the packets it actually offers.

Primary references consulted (2026-10-27):

- [Freeciv helpdata](https://github.com/freeciv/freeciv/blob/main/data/helpdata.txt)
  and the [HOWTOPLAY documentation](https://github.com/freeciv/freeciv/blob/main/doc/HOWTOPLAY)
  for food, production, city growth, terrain improvements, research, combat,
  exploration, and fog-of-war concepts.
- [Freeciv classic ruleset data](https://github.com/freeciv/freeciv/tree/main/data/classic)
  for the fact that unit, technology, terrain, and production details are
  ruleset data rather than safe fixed prompt constants.
- [ReAct](https://arxiv.org/abs/2210.03629),
  [Reflexion](https://arxiv.org/abs/2303.11366), and
  [Voyager](https://arxiv.org/abs/2305.16291) for closed-loop observation,
  action, feedback, and compact memory patterns in interactive agents.

These sources motivate survival-first conditional heuristics, explicit
uncertainty, and a short feedback-oriented memory. They do not establish the
legal action surface, candidate IDs, research costs, combat odds, victory
settings, or available abilities for this harness. The current snapshot,
candidate table, and execution feedback therefore take precedence; no source
supports asking the model to emit hidden chain-of-thought.

Local evidence traceability:

- Candidate whitelist and ID normalization: `src/lechonk/adapter/candidates.py`.
- Candidate/action target descriptors and execution semantics:
  `src/lechonk/core/runner.py` (`_describe`, `_execute_one`).
- Commitment target and verb matching: `src/lechonk/core/metrics.py`
  (`_target_match`, `_verb_compatible`).
- ActionBatch and typed commitment schema: `src/lechonk/core/schemas.py`.
- Prompt assembly and repair dispatch: `src/lechonk/model/prompts.py` and
  `src/lechonk/model/turn_client.py`.
