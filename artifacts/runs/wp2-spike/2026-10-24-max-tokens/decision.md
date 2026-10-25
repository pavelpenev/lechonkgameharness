# FW-05 decision record — DEFAULT_MAX_TOKENS re-examination (offline)

- Date: 2026-10-24. Package: FW-05 (plan §5, harness-fixes-v1). Status: DECIDED — raise the cap to 16384 (provisional; re-check after FW-22). Re-check scheduled after FW-22's re-spike.
- Evidence (read-only): `artifacts/runs/wp4-eval3/p1/arms/model/api_calls.jsonl` (89 records) and `artifacts/runs/wp4-eval3/p2/arms/model/api_calls.jsonl` (91 records), parsed with the repo `.venv/bin/python`. No existing artifact was modified; no src/ file touched.
- Code point of decision: `DEFAULT_MAX_TOKENS = 8192` (`src/lechonk/model/mistral_client.py:82` at commit 6939f3e).

## 1. Finish-reason distribution (all attempts, incl. the 1 fingerprint probe per arm)

| finish_reason | p1 (n=89) | p2 (n=91) |
|---|---|---|
| stop | 33 | 44 |
| length | 35 | 47 |
| null (transport read-timeout, no response) | 21 | 0 |

Attempt kinds: p1 = 1 probe / 50 initial / 38 repair; p2 = 1 probe / 50 initial / 40 repair. Outcomes: p1 = ok 32, schema_error 35, transport_error 21, probe 1; p2 = ok 43, schema_error 47, probe 1.

## 2. tokens_out histogram, capped vs uncapped (initial + repair attempts)

- **Capped (tokens_out == 8192):** p1 35 calls, p2 47 calls — exactly the README:453-455 figures (40% / 52% of turn calls). Every capped call also carries `finish_reason: "length"` and outcome `schema_error` (truncated JSON cannot parse).
- **Uncapped (finish stop):** p1 33 calls spanning 1872–7992 tokens_out (mean 5177.0); p2 44 calls spanning 1873–8039 (mean 5052.4). No uncapped completion reached the cap; the observed maximum is 8039 < 8192.
- tokens_out 0 records: p1's 21 transport-error attempts (no response). Full per-value histogram: every uncapped value occurs exactly once (continuous distribution, no clustering at the cap other than the 8192 spike).

## 3. Capped-turn outcomes (schema_error → repair → degraded chain)

Turns where the **initial** attempt hit `length`: p1 32/50, p2 40/50. Turn-level resolution of those turns:

| resolution of an initial-capped turn | p1 | p2 |
|---|---|---|
| repair attempt finished `stop`, outcome `ok` (turn recovered) | 19 | 33 |
| repair attempt ALSO hit `length` (both attempts capped → turn degraded) | 3 (turns 12, 13, 31) | 7 (turns 17, 19, 34, 35, 36, 37, 47) |
| repair attempt transport-timeout (turn degraded, not from capping) | 10 | 0 |

Both-attempts-capped turns, verified chain per record: initial `length` @8192 outcome `schema_error` → repair `length` @8192 outcome `schema_error` → turn fell to the degraded default (per-arm example records: p1 turn 12 call1/call2 at $0.015908/$0.017555; p2 turn 17 call1/call2 at $0.016442/$0.018298). This matches README limitation 2 (3 turns p1, 7 turns p2).

## 4. Repair recovery rate for capped turns

- p1: 19/32 = **59.4%** (and 10 further turns were lost to transport timeouts before capping could even be resolved).
- p2: 33/40 = **82.5%**.
- Capped-but-repaired turns still pay a full second call: observed repair-call cost ≈ $0.017–0.021 each and repair latencies dominated the per-turn wall clock (mean capped-call latency p1 84.6 s, p2 56.4 s; overall mean 65.9 s / 45.2 s).

## 5. Cost/latency deltas of candidate caps — BOUNDED ESTIMATES

**Pricing (derived from the logs themselves, not assumed):** least-squares fit of `cost_usd = a·tokens_in + b·tokens_out` over the 159 priced, completed calls (max residual 5.5e-7 USD) gives **$0.50/M input, $1.50/M output** — consistent with the README:402 recorded $2.45 total for 1,744,241 in / 1,055,018 out.

**Why these are bounds, not measurements.** `finish_reason: "length"` proves a truncated call wanted *more* than 8192 output tokens but the log cannot observe the true uncapped length. Additionally, these runs pre-date the plan's sampling change (they ran at temperature 0.0 with no reasoning effort; FW-20 moves to temperature 1.0 with `reasoning_effort="high"`, whose reasoning tokens also draw on the completion budget) — so the distributions below are expected to shift. The decision is therefore **provisional and scheduled for re-check after FW-22's re-spike** (plan §5 FW-22).

**Anchor for the central estimate:** on capped turns, the repair attempt (same turn, same task, fresh 8192 budget) finished `stop` at 2995–6733 tokens (p1, mean 5177) and 1873–8039 (p2, mean 5052) — an empirical hint that roughly ~5k further output tokens of content were outstanding after a truncation, though a repair re-emits the full response and thus over-states the true residual.

| scenario | extra output tokens | extra cost (p1+p2) | extra latency |
|---|---|---|---|
| **Lower bound** (capped completions stop immediately above 8192) | 0 | $0.00 | +0 s |
| **Central** (repair-anchor: +5177 p1 / +5052 p2 per capped call, capped at the new cap) | ~181k (p1) + ~202k (p2) ≈ 383k | ~$0.57 | ~+32 min |
| **Upper bound** (every capped call uses the full additional headroom: +8192 each) | 286,720 (p1) + 385,024 (p2) = 671,744 | ~$1.01 | ~+49 min (p1) + ~44 min (p2) |

Both candidate caps (16384, 24576) give **identical** lower/central estimates because no observed completion plausibly exceeds 16384 (longest uncapped completion: 8039; even the repair-anchor residual of ~5k on top of 8192 lands at ~13.3k < 16384). They differ only in worst-case exposure (24576 admits up to +16384 per capped call, unbounded by any observation).

**Offsetting savings not netted above:** raising the cap eliminates the repair call on recovered turns (p1 19 + p2 33 = 52 repairs × ≈ $0.017 ≈ **$0.90 saved**) plus their latency, and converts the 10 both-capped degraded turns (3 p1, 7 p2) into normal turns. Net cost of the raise is therefore likely near zero to modestly negative; the real cost is latency (single long calls vs two shorter ones — roughly a wash) and the risk of longer rambles at temperature 1.0.

## 6. Decision

**Raise `DEFAULT_MAX_TOKENS` from 8192 to 16384.**

Rationale:
1. Truncation is the single largest measured failure mode: 40% (p1) / 52% (p2) of turn calls hit the cap, every one producing a `schema_error`, triggering a repair call or a degraded turn. This dwarfs transport errors as a self-inflicted failure source.
2. The cap is not just reached, it is *binding*: 3 (p1) and 7 (p2) turns exhausted both attempts at the cap and degraded; repair recovery is imperfect (59%–82%) and p1's recovery was additionally eroded by transport timeouts.
3. Bounded cost of the raise is small (≤ ~$1.01/run worst case, likely offset by eliminated repair calls) and no observed completion suggests content beyond 16384.
4. 16384 over 24576: identical expected behavior at half the worst-case latency/cost exposure; 24576 has no supporting observation.
5. The new sampling regime (temperature 1.0, `reasoning_effort="high"`) plausibly *increases* completion demand (reasoning tokens share the completion budget), arguing against keeping the tight cap.

**Provisional caveat:** recorded evidence pre-dates the sampling change and cannot measure uncapped lengths. **This decision MUST be re-checked after FW-22's re-spike** (plan §5: "The package also re-checks the FW-05 max_tokens decision against the new sampling's finish_reason distribution"); if the re-spike shows materially different capping behavior, a new dated decision record supersedes this one. If confirmed, FW-05b (conditional package, chain M after FW-21) applies the constant with its own acceptance (`pytest tests/test_mistral_client.py -q` green with the new default asserted).

## 7. Assumptions

- Pricing fit assumed linear in/out token cost and used only completed (non-null `finish_reason`) priced records; the fit residual (< 6e-7 USD) validates this on these logs.
- The repair-attempt token profile is treated as an upper-flavored proxy for outstanding content after truncation (it re-emits the whole response, so it likely over-estimates the true residual).
- Latency assumed roughly proportional to output tokens within a call (observed capped calls are the slowest population in both arms).
- The histogram reflects temperature-0.0, no-reasoning-effort sampling; transfer to the temperature-1.0 / high-reasoning regime is unverified until FW-22.

*Computed and recorded by FW-05 (offline analysis only; no live provider calls, no src/ changes, no existing artifacts modified).*
