"""WP2 enforcement spike (design-plan-freeciv-v1.1.md decision 4, probe P10).

Resolves MISTRAL_API_KEY via the narrow loader (``load_api_key_from_env_file``):
the process environment first, then the repository-root ``.env``. Only the
presence boolean is ever printed or recorded — never the value.

If present: live-probes the exact model id ``mistral-large-4`` (P10a: the ID
responds — no silent substitution) and Mistral ``json_schema``
structured-output support on it (P10b: a minimal ActionBatch-shaped request
carrying the real ActionBatch JSON schema, whose actions[].ref is
constrained to candidate IDs), records prompt / schema hash / raw response
shape / verdict under artifacts/runs/wp2-spike/, runs the per-arm model
fingerprint probe (decision 8), and writes the spike decision (json_schema
vs schema-validated parsing) shaped for the run manifest.

If absent: writes the structured blocker (missing evidence), verifies the
client offline against a mock transport (both enforcement paths, the
attempt-cap contract, per-call logging fields, the no-secret assertion),
and records the provisional default (validated_parse) pending the live probe.

Exit codes: 0 = live probe done, decision recorded; 1 = live probe ran and
FAILED (model id or json_schema unsupported — no silent substitution, model
dependent work stops); 3 = blocked (key absent), offline mock evidence
recorded.
"""
import inspect
import json
import os
import sys
from datetime import datetime, timezone

from lechonk.core.schemas import CANDIDATE_ID_REGEX
from lechonk.model import mistral_client as mc
from lechonk.model import prompts

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR = os.path.join(REPO, 'artifacts', 'runs', 'wp2-spike')
os.makedirs(RUNDIR, exist_ok=True)

LOG_PATH = os.path.join(RUNDIR, 'wp2_spike_stdout.log')
_log_file = None


def log(msg):
    """Print to stdout AND persist to the run log (HANDOFF: persist run
    stdout/logs as artifacts under artifacts/runs/)."""
    global _log_file
    if _log_file is None:
        _log_file = open(LOG_PATH, 'a', encoding='utf-8')
    line = f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {msg}'
    print(line, flush=True)
    _log_file.write(line + '\n')
    _log_file.flush()


def write_json(name, obj):
    path = os.path.join(RUNDIR, name)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write('\n')
    log(f'wrote {name}')
    return path


# The fixed P10b probe prompt (a minimal ActionBatch-shaped request). The
# schema itself is the real core.schemas.ActionBatch JSON schema — single
# source of truth, derived from the pydantic model, never hand-maintained.
P10_PROBE_PROMPT = (
    'Return a valid ActionBatch for turn t1: exactly one action referencing '
    'the candidate ID unit:1:goto_n, no commitments, finish_requested true, '
    'and a one-line note. Follow the schema exactly.')


def probe_request_plan():
    """The exact P10 request plan (no secrets): what would be sent to the
    Studio API, recorded whether or not the key is present."""
    return {
        'probe': 'P10',
        'model_id': mc.MODEL_ID,
        'substitution_note': (
            'mistral-large-latest does NOT point at mistral-large-4; no '
            'silent substitution (decision 4)'),
        'p10a_model_id_check': {
            'messages': [{'role': 'user', 'content': 'Reply with exactly: OK'}],
            'temperature': 0.0,
            'max_tokens': 16,
            'response_format': None,
        },
        'p10b_json_schema_check': {
            'prompt': P10_PROBE_PROMPT,
            'temperature': 0.0,
            'max_tokens': 1024,
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'action_batch',
                    'schema_definition_hash': mc.action_batch_schema_hash(),
                    'strict': True,
                },
            },
            'schema_source': (
                'core.schemas.ActionBatch.model_json_schema(); actions[].ref '
                'is constrained to the candidate-ID regex'),
        },
        'schema_hash': mc.action_batch_schema_hash(),
        'prompt_hashes': prompts.prompt_manifest_hashes(),
    }


# ---------------------------------------------------------------------------
# Offline mock verification (key absent path)
# ---------------------------------------------------------------------------

def offline_mock_verification():
    """Drive the client against a mock transport: both enforcement paths,
    the attempt-cap contract (≤2 attempts/turn, ≤1 repair), per-call logging
    fields, the no-secret assertion, and the deterministic fingerprint probe.
    Returns an evidence dict (all checks with pass/fail)."""
    from dataclasses import dataclass, field

    canary = 'sk-CANARY-9f3a91c2-do-not-leak'
    valid_batch = json.dumps({
        'turn_id': 'spike/t1',
        'actions': [{'ref': 'unit:1:goto_n'}],
        'commitments': [],
        'finish_requested': True,
        'notes': 'probe batch',
    })

    @dataclass
    class MockTransport:
        outcomes: list = field(default_factory=list)
        calls: list = field(default_factory=list)

        def complete(self, *, messages, model, temperature, max_tokens,
                     response_format, random_seed, reasoning_effort=None,
                     n=None, timeout_ms=None,
                     disable_transport_retries=False):
            self.calls.append({
                'messages': messages, 'model': model,
                'temperature': temperature, 'max_tokens': max_tokens,
                'response_format': response_format,
                'reasoning_effort': reasoning_effort, 'n': n,
            })
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    def raw(content, **kw):
        return mc.RawCompletion(
            content=content, model=mc.MODEL_ID,
            response_id=kw.get('response_id', 'resp-mock'), created=0,
            finish_reason=kw.get('finish_reason', 'stop'),
            prompt_tokens=kw.get('prompt_tokens', 100),
            completion_tokens=kw.get('completion_tokens', 50),
            total_tokens=(kw.get('prompt_tokens', 100)
                          + kw.get('completion_tokens', 50)),
            transport_retries=kw.get('transport_retries', 0))

    messages = [
        {'role': 'system', 'content': prompts.system_prompt()},
        {'role': 'user', 'content': prompts.turn_prompt(
            memory='mem', snapshot='snap', delta='delta',
            candidate_summary='unit:1:goto_n — move north')},
    ]
    checks = {}

    # 1. validated_parse: parses a valid batch; json_object response format
    t = MockTransport(outcomes=[raw(valid_batch)])
    c = mc.MistralClient(api_key=canary, transport=t)
    res = c.action_batch(messages=messages, prompt_hash='sha256:spike',
                         attempts=mc.TurnAttempts())
    checks['validated_parse_path'] = (
        res.batch.turn_id == 'spike/t1'
        and res.attempts_used == 1 and res.repairs_used == 0
        and t.calls[0]['response_format'] == {'type': 'json_object'})

    # 2. json_schema path: strict structured-output request with the real
    #    ActionBatch schema (candidate-ID-constrained refs). Pydantic's
    #    JSON-schema export renders the ref pattern in ECMA flavor
    #    (\Z -> \z), so the conjunct compares the ECMA-translated regex.
    ecma_ref_pattern = CANDIDATE_ID_REGEX.replace('\\Z', '\\z')
    t = MockTransport(outcomes=[raw(valid_batch)])
    c = mc.MistralClient(api_key=canary, transport=t,
                         enforcement='json_schema')
    c.action_batch(messages=messages, prompt_hash='sha256:spike',
                   attempts=mc.TurnAttempts())
    rf = t.calls[0]['response_format']
    schema = rf['json_schema']['schema_definition']
    checks['json_schema_path'] = (
        rf['type'] == 'json_schema'
        and rf['json_schema']['name'] == 'action_batch'
        and rf['json_schema']['strict'] is True
        and schema == mc.action_batch_json_schema()
        and schema['$defs']['ActionItem']['properties']['ref']['pattern']
        == ecma_ref_pattern)

    # 3. attempt cap: garbage on both attempts -> AttemptBudgetExhausted,
    #    exactly 2 transport calls
    t = MockTransport(outcomes=[raw('garbage'), raw('garbage')])
    c = mc.MistralClient(api_key=canary, transport=t)
    try:
        c.action_batch(messages=messages, prompt_hash='sha256:spike',
                       attempts=mc.TurnAttempts())
        checks['attempt_cap_enforced'] = False
    except mc.AttemptBudgetExhausted:
        checks['attempt_cap_enforced'] = len(t.calls) == 2

    # 4. repair counting: garbage then valid -> 2 attempts, 1 repair, and
    #    the repair messages carry the class-specific repair prompt
    t = MockTransport(outcomes=[raw('garbage'), raw(valid_batch)])
    c = mc.MistralClient(api_key=canary, transport=t)
    res = c.action_batch(
        messages=messages, prompt_hash='sha256:spike',
        attempts=mc.TurnAttempts(),
        repair_candidates_for=lambda rr: 'FULL TABLE: unit:1:goto_n')
    repair_user = t.calls[1]['messages'][-1]['content']
    checks['repair_counted_and_class_prompt'] = (
        res.attempts_used == 2 and res.repairs_used == 1
        and res.call_logs[1].kind == 'repair'
        and len(t.calls[1]['messages']) == len(messages) + 2
        and 'SCHEMA ERROR' in repair_user
        and 'FULL TABLE: unit:1:goto_n' in repair_user)

    # 5. per-call logging fields (tokens, latency, cost, fingerprint data)
    t = MockTransport(outcomes=[raw(valid_batch, prompt_tokens=100,
                                    completion_tokens=50)])
    c = mc.MistralClient(api_key=canary, transport=t)
    c.action_batch(messages=messages, prompt_hash='sha256:spike',
                   attempts=mc.TurnAttempts())
    entry = c.call_logs[0].to_dict()
    expected_cost = round((100 * 0.50 + 50 * 1.50) / 1_000_000, 6)
    checks['per_call_logging_fields'] = (
        entry['tokens_in'] == 100 and entry['tokens_out'] == 50
        and entry['tokens_total'] == 150
        and isinstance(entry['latency_ms'], int)
        and entry['cost_usd'] == expected_cost
        and entry['model_requested'] == mc.MODEL_ID
        and entry['model_responded'] == mc.MODEL_ID
        and entry['fingerprint'] == 'unprobed'
        and entry['finish_reason'] == 'stop'
        and entry['response_id'] == 'resp-mock')

    # 6. no-secret: the canary key never appears in any serialized log
    t = MockTransport(outcomes=[raw(valid_batch), raw('garbage'),
                                raw(valid_batch)])
    c = mc.MistralClient(api_key=canary, transport=t)
    c.action_batch(messages=messages, prompt_hash='sha256:spike',
                   attempts=mc.TurnAttempts())
    c.action_batch(messages=messages, prompt_hash='sha256:spike',
                   attempts=mc.TurnAttempts())
    blob = json.dumps([l.to_dict() for l in c.call_logs])
    checks['no_secret_in_logs'] = canary not in blob

    # 7. fingerprint probe: K=3 multi-sample, arm sampling (temperature
    #    1.0, reasoning_effort "high", no top_p), set-encoded fingerprint
    t = MockTransport(outcomes=[raw('LECHONK-FINGERPRINT-7f3a91c2')])
    c = mc.MistralClient(api_key=canary, transport=t)
    p = c.probe_fingerprint()
    import hashlib
    expected = 'sha256:' + hashlib.sha256(
        b'LECHONK-FINGERPRINT-7f3a91c2').hexdigest()
    checks['fingerprint_probe_mock'] = (
        p.fingerprint == expected
        and t.calls[0]['temperature'] == 1.0
        and t.calls[0]['reasoning_effort'] == 'high'
        and t.calls[0]['n'] == mc.FINGERPRINT_PROBE_SAMPLES
        and 'top_p' not in t.calls[0]
        and p.prompt_hash == prompts.prompt_hash(
            mc.FINGERPRINT_PROBE_PROMPT)
        and c.arm_fingerprint == expected)

    return {
        'checks': checks,
        'all_passed': all(checks.values()),
        'mock_transport': 'script-local MockTransport (no network, no SDK import)',
        'canary_key_used': 'sk-CANARY-... (test canary, never a real key)',
    }


def main():
    # Presence check ONLY — the value is never read into logs or artifacts.
    # The narrow loader (load_api_key_from_env_file) resolves MISTRAL_API_KEY
    # from the process environment or, failing that, the repository-root
    # .env; only the presence boolean below is ever recorded.
    try:
        mc.load_api_key_from_env_file()
    except mc.MistralKeyError:
        pass  # recorded below as the structured blocker
    key_present = os.environ.get('MISTRAL_API_KEY') is not None
    log(f'wp2-spike start: MISTRAL_API_KEY present in process environment: '
        f'{key_present}')
    log('  (only the presence boolean is recorded; the key value is never '
        'printed or persisted; when absent from the environment the key was '
        'loaded from the repo-root .env by the narrow loader)')

    plan = probe_request_plan()
    write_json('probe_request_plan.json', plan)
    log(f'  probe plan: model={plan["model_id"]} '
        f'schema_hash={plan["schema_hash"]}')

    if key_present:
        client = mc.MistralClient()  # env key, exact model id
        p10a = client.probe_model_id()
        log(f'  P10a model-id check: responded={p10a.responded} '
            f'model_responded={p10a.model_responded} error={p10a.error}')
        p10b = client.probe_json_schema_support(probe_prompt=P10_PROBE_PROMPT)
        log(f'  P10b json_schema check: responded={p10b.responded} '
            f'supported={p10b.json_schema_supported} '
            f'shape={json.dumps(p10b.response_shape, sort_keys=True)}')
        fingerprint = client.probe_fingerprint()
        log(f'  fingerprint probe: {fingerprint.fingerprint}')
        write_json('p10_live_probe.json', {
            'p10a_model_id': p10a.to_dict(),
            'p10b_json_schema': p10b.to_dict(),
            'fingerprint_probe': fingerprint.to_dict(),
            'probe_prompt': P10_PROBE_PROMPT,
            'schema_hash': mc.action_batch_schema_hash(),
        })
        write_json('fingerprint_probe.json', fingerprint.to_dict())
        supported = p10a.responded and p10b.json_schema_supported
        decision = mc.spike_decision(
            live_probe_done=True,
            json_schema_supported=p10b.json_schema_supported)
        report = {
            'probe': 'P10',
            'model_id': mc.MODEL_ID,
            'date': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'verdict': 'decided_live' if supported else 'live_probe_failed',
            'evidence': 'p10_live_probe.json (prompt, schema hash, raw '
                        'response shape, verdict — no secrets, no raw model '
                        'output beyond the shape evidence)',
            'p10a_model_id_responded': p10a.responded,
            'p10b_json_schema_supported': p10b.json_schema_supported,
            'p10b_response_shape': p10b.response_shape,
            'schema_hash': mc.action_batch_schema_hash(),
            'probe_prompt': P10_PROBE_PROMPT,
            'decision': decision,
            'failure_response': (
                'No silent substitution: model-dependent work stops, user '
                'notified; no other model is used (P10 failure response).'
                if not supported else None),
        }
        write_json('spike_report.json', report)
        log(f'  verdict: {report["verdict"]}; decision: '
            f'{decision["decision"]} ({decision["decision_status"]})')
        _log_file.close()
        return 0 if supported else 1

    # ---- key absent: structured blocker + offline mock verification ----
    evidence = offline_mock_verification()
    write_json('offline_mock_evidence.json', evidence)
    log(f'  offline mock verification: all_passed={evidence["all_passed"]} '
        f'checks={json.dumps(evidence["checks"], sort_keys=True)}')
    decision = mc.spike_decision()  # provisional (live probe blocked)
    report = {
        'probe': 'P10',
        'model_id': mc.MODEL_ID,
        'date': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'verdict': 'blocked',
        'blocker': {
            'kind': 'missing_evidence',
            'missing': 'No MISTRAL_API_KEY in the process environment and '
                       'no MISTRAL_API_KEY line in the repo-root .env (the '
                       'narrow loader tried both); the live P10 probe '
                       'cannot run',
            'evidence_needed': (
                'live probe of the exact model id mistral-large-4: (a) the '
                'ID responds, (b) json_schema structured outputs with the '
                'real ActionBatch schema (candidate-ID-constrained refs) '
                'parse as an ActionBatch'),
            'loader_tried': (
                'the narrow loader (load_api_key_from_env_file) already '
                'resolved MISTRAL_API_KEY from the process environment or '
                'the repository-root .env before this blocker; neither '
                'source provided a key'),
        },
        'provisional_decision': decision,
        'offline_mock_evidence': 'offline_mock_evidence.json',
        'probe_request_plan': 'probe_request_plan.json (the exact P10 request '
                              'that would be sent, with schema hash)',
        'schema_hash': mc.action_batch_schema_hash(),
        'probe_prompt': P10_PROBE_PROMPT,
        'remedy': (
            'export MISTRAL_API_KEY in the process environment (or add an '
            'MISTRAL_API_KEY=... line to the repository-root .env) and '
            're-run scripts/wp2_spike.py; the live probe and the spike '
            'decision (json_schema vs schema-validated parsing) then '
            'complete and the decided default is written to the run '
            'manifest'),
    }
    write_json('spike_report.json', report)
    log(f'  verdict: blocked; provisional decision: {decision["decision"]} '
        f'({decision["decision_status"]})')
    log('  structured blocker recorded in spike_report.json; offline mock '
        'verification recorded in offline_mock_evidence.json')
    _log_file.close()
    return 3


def respike_main() -> int:
    """FW-22 dated re-spike: re-run the P10a model-id probe, the P10b
    json_schema support probe, and the K=3 fingerprint probe LIVE under the
    new arm sampling (temperature 1.0, reasoning_effort "high", no top_p),
    into a NEW dated dir artifacts/runs/wp2-spike/<execution-date>-respike/.
    Existing artifacts are never touched; probe failures are reported,
    never faked."""
    global RUNDIR, LOG_PATH
    execution_date = datetime.now().astimezone().date().isoformat()
    outdir = os.path.join(REPO, 'artifacts', 'runs', 'wp2-spike',
                          f'{execution_date}-respike')
    os.makedirs(outdir, exist_ok=True)
    RUNDIR = outdir
    LOG_PATH = os.path.join(outdir, 'wp2_spike_stdout.log')

    try:
        client = mc.MistralClient()  # env key, exact pinned model id
    except mc.MistralKeyError as exc:
        write_json('respike_report.json', {
            'probe': 'FW-22 re-spike',
            'model_id': mc.MODEL_ID,
            'execution_date': execution_date,
            'verdict': 'blocked',
            'blocker': {'kind': 'missing_evidence', 'detail': str(exc)},
        })
        log(f'respike blocked: {exc}')
        return 4

    p10a = client.probe_model_id()
    p10b = client.probe_json_schema_support(probe_prompt=P10_PROBE_PROMPT)
    fingerprint = client.probe_fingerprint()
    probe_call_logs = [
        {'call_id': l.call_id, 'kind': l.kind, 'temperature': l.temperature,
         'reasoning_effort': l.reasoning_effort, 'tokens_in': l.tokens_in,
         'tokens_out': l.tokens_out, 'tokens_total': l.tokens_total,
         'latency_ms': l.latency_ms, 'finish_reason': l.finish_reason,
         'model_requested': l.model_requested,
         'model_responded': l.model_responded}
        for l in client.call_logs]
    try:
        from mistralai.client.chat import Chat
        sdk_chat_params = sorted(
            inspect.signature(Chat.complete).parameters)
    except Exception:
        sdk_chat_params = None
    supported = p10a.responded and p10b.json_schema_supported
    report = {
        'probe': 'FW-22 re-spike',
        'model_id': mc.MODEL_ID,
        'execution_date': execution_date,
        'session_calendar_note': (
            'the session calendar said 2026-10-24; the host date at '
            f'execution ({execution_date}) is used'),
        'sampling': {
            'temperature': client.temperature,
            'reasoning_effort': client.reasoning_effort,
            'max_tokens_default': mc.DEFAULT_MAX_TOKENS,
            'top_p_sent': False,
            'transport_parameter_names': sorted(
                inspect.signature(
                    mc.MistralSDKTransport.complete).parameters),
            'sdk_chat_parameter_names': sdk_chat_params,
        },
        'probes': {
            'p10a_model_id': p10a.to_dict(),
            'p10b_json_schema': p10b.to_dict(),
            'fingerprint_k3': fingerprint.to_dict(),
        },
        'probe_call_logs': probe_call_logs,
        'max_tokens_recheck': {
            'scope_note': (
                'probe-scale only: these are individual probe calls '
                '(max_tokens 16 / 1024 / 1024), not full turns; a full '
                'turn-scale finish_reason re-check requires a future live '
                'run under the same arm sampling'),
            'p10a_model_id': {
                'finish_reason': p10a.finish_reason,
                'tokens_in': None, 'tokens_out': None,
            },
            'p10b_json_schema': {
                'finish_reason': p10b.response_shape.get('finish_reason'),
                'tokens_in': None, 'tokens_out': None,
            },
            'fingerprint_k3': {
                'finish_reason': fingerprint.finish_reason,
                'tokens_in': fingerprint.tokens_in,
                'tokens_out': fingerprint.tokens_out,
                'tokens_total': fingerprint.tokens_total,
                'sample_attempts': fingerprint.sample_attempts,
                'sample_mode': fingerprint.sample_mode,
                'output_hashes_unique': len(fingerprint.output_hashes),
            },
        },
    }
    write_json('respike_report.json', report)
    log(f'respike verdict: '
        f'{"decided_live" if supported else "live_probe_failed"}; '
        f'p10a.responded={p10a.responded} '
        f'p10b.json_schema_supported={p10b.json_schema_supported}')
    _log_file.close()
    return 0 if supported else 1


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'respike':
        sys.exit(respike_main())
    sys.exit(main())
