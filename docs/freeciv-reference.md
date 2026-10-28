# Freeciv reference data

The harness exposes reference material through the model's bounded,
read-only `lookup_freeciv` tool. It is supporting documentation, not hidden
game state and not a replacement for the live observation or candidate table.

## What is provisioned

Provisioning attempts two independent optional extractions from the verified
Freeciv-Web container:

1. `scripts/extract_freeciv_reference.sh` copies the `classic` ruleset files
   and, when the image has it, the static `helpdata.txt` file to
   `artifacts/reference/freeciv/`.
2. `scripts/extract_freeciv_manual.sh` runs Freeciv's installed
   `freeciv-manual --ruleset classic --wiki` generator and copies its eight
   generated MediaWiki files (`classic1.mediawiki` through
   `classic8.mediawiki`) to `artifacts/reference/freeciv-manual/`.

The current pinned repository image is Freeciv `3.2.91-dev` at
`civrealm/freeciv-web@sha256:4923253876a76d3c5f55c6007e530f6c3cd3120b9e2f37a8e943faf920cb373a`.
That particular image does not contain `helpdata.txt`, but does contain the
manual generator. This observation is deliberately scoped to the pinned image;
other Freeciv images may provide different files or paths.

Manual extraction is optional and can be run independently:

```sh
./scripts/extract_freeciv_manual.sh
```

Set `LECHONK_EXTRACT_FREECIV_MANUAL=0` to skip the automatic provisioner step,
or `FREECIV_MANUAL_REQUIRED=1` to make an unavailable generator fatal. The
generic extractor has analogous `LECHONK_EXTRACT_FREECIV_REFERENCE=0`,
`FREECIV_REFERENCE_REQUIRED=1`, and `FREECIV_REFERENCE_REQUIRE_HELP=1`
controls. Existing artifacts are verified by image identity and SHA-256; a
stale or mismatched directory is not overwritten. Move it aside and rerun the
extractor instead of changing a manifest hash manually.

Each artifact directory contains a `reference.json` provenance manifest and a
`LICENSE.txt`. The generated manual is derived from Freeciv data, which is
GPL-2.0-or-later; retain upstream notices before redistributing it. The
manifest records the image, ruleset, generator, file hashes, and manual format.
For the current pinned image, the extracted classic ruleset tree hash is
recorded as `406610064a5c8b3d1f95ddb039635e30c9d44b0e204b6347a4060c0cad643d3f`.

## Lookup precedence

For a query during a live turn, the harness searches in this order:

1. The current live `RulesetCtrl` snapshot (the authoritative active rules).
2. The extracted generated classic manual, if available.
3. `helpdata.txt`, if available.
4. The extracted classic ruleset files, if available.

The first source is preferred for names and mechanics because it comes from the
running session. Static results include source/provenance labels and hashes in
run metadata. Missing sources are reported as unavailable; the harness does not
fetch documentation from the network during a game.

The model can request only bounded searches: the query is limited to 256
characters, a lookup returns at most eight entries and 4,000 rendered
characters, and a turn has a small fixed lookup-call budget. The tool is
read-only. Current fog-safe state, legal candidates, and server read-backs
always override historical or static prose.

## Manual subset and limitations

The generated MediaWiki source is retained as eight files, but the runtime
index intentionally extracts only bounded, plain-text entry sections. The
current pinned artifact indexes 449 entries across options, technologies,
commands, units, buildings, terrain, wonders, and governments. It does not
evaluate MediaWiki templates, execute links, reconstruct generated tables, or
claim that every sentence in the source is indexed. It cannot answer current
map, visibility, ownership, diplomacy, production, or legal-action questions.

`helpdata.txt` is a separate optional source. Its absence in the pinned image
is normal and is recorded as `helpdata_status: "unavailable"`; it does not mean
that the generated manual failed. Use
`scripts/extract_freeciv_reference.sh --diagnose` to inspect a different image
before changing paths or requirements.
