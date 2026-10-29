# Narrated video guide

This is a local, evidence-linked post-production workflow. Keep the raw capture
images, event logs, storyboard, generated WAVs, and manifests; the rendered video
is a derived artifact. Paths in the examples are interpreted relative to the
directory from which each command is run.

## 1. Generate narration

`generate_narration.py` takes a positional JSON path. The file must be a
non-empty list of objects with unique string `id` and `text` fields:

```json
[
  {"id": "intro", "text": "Le Chonk builds a civilization from a fogged map."},
  {"id": "turn-51", "text": "At turn fifty-one, the resumed game continues."}
]
```

Run it with a new output directory:

```sh
uv run python scripts/generate_narration.py segments.json --output-dir narration
```

The defaults are model `voxtral-mini-tts-3-0`, `voice_id=voice_592` (Marie,
French), and `expressiveness=1`. The script loads `.env` through the project's
environment loader; do not put secrets in this document or in committed input
files. It makes no paid retries. Existing WAVs are reused only when the request
fingerprint and manifest hash match, so preserve `narration/manifest.json` and
the WAVs together.

## 2. Render the storyboard

`render_narrated_story.py` requires `--storyboard` and `--output-dir`; the output
directory must be new:

```sh
uv run python scripts/render_narrated_story.py \
  --storyboard storyboard.json --output-dir rendered-story
```

The storyboard is an object containing a non-empty `scenes` array. Each scene
has `kind` (`intro`, `gameplay`, or `outro`) and a positive `duration`; gameplay
scenes additionally need an existing `image`. Optional `audio` files are padded
with 0.8 seconds by the renderer, so the total encoded duration (including
intro/outro and audio padding) must be at most 120 seconds. A minimal valid
storyboard is:

```json
{
  "scenes": [
    {"id": "intro", "kind": "intro", "duration": 3,
     "title": "LE CHONK", "caption": "A fogged map. One long game."},
    {"id": "turn-51", "kind": "gameplay", "duration": 8,
     "image": "captures/turn-51.png", "crop": [0, 0, 1530, 1080],
     "turn": 51, "title": "The resumed frontier", "caption": "Evidence-linked still frame.",
     "evidence": {"run": "artifacts/runs/lechonk-none-recording-100-20261029"}},
    {"id": "outro", "kind": "outro", "duration": 3,
     "title": "A recorded experiment", "caption": "Mechanics first; claims bounded."}
  ]
}
```

`crop` is explicit source-image pixels in `[left, top, right, bottom]` order;
it is bounds-checked before rendering. The final 118-second storyboard used
scene durations `12/15/16/16/16/16/15/12` seconds. Keep source and evidence
references in the storyboard/provenance so every still can be traced to the
capture. The renderer and capture schema are described in
[`docs/gameplay-capture-design.md`](gameplay-capture-design.md); the latest
recording evidence is under the ignored local run directory
`artifacts/runs/lechonk-none-recording-100-20261029/`.

## 3. Music post-process

Music is a recorded FFmpeg stage, not a checked-in mixer script. The release
mix reuses the selected Musopen recording with attribution to the Skidmore
College Orchestra. It takes excerpt seconds 40–158, applies gain `-27 dB`,
one-second fade-in and three-second fade-out, and ducks music under narration
with threshold `0.02`, ratio `6`, attack `20 ms`, and release `500 ms`. Credits
start at 106 seconds. The exact selected track is the recording at
<https://commons.wikimedia.org/wiki/File:Gustav_Holst_-_the_planets,_op._32_-_i._mars,_the_bringer_of_war.ogg>.
The exact local command description and selected-source
record are preserved in [`filtergraph.txt`](../artifacts/highlights/lechonk-demo-118-louder-20261029/filtergraph.txt),
[`provenance.json`](../artifacts/highlights/lechonk-demo-118-louder-20261029/provenance.json),
and [`copiedcredits.txt`](../artifacts/highlights/lechonk-demo-118-louder-20261029/copiedcredits.txt).

Do not assume a clean clone can reproduce the finished mix: it requires
user-provided source captures, generated audio, and the selected music input.
Do not delete raw assets after rendering; they are the evidence and recovery
path for an editable story.

## 4. Verify and preserve

Probe the final file and perform a full decode before delivery:

```sh
ffprobe -v error -show_format -show_streams artifacts/highlights/lechonk-demo-118-louder-20261029/story.mp4
ffmpeg -v error -i artifacts/highlights/lechonk-demo-118-louder-20261029/story.mp4 -f null -
```

Keep the MP4 beside its provenance, filtergraph, credits, storyboard, narration
manifest/WAVs, and untouched raw capture directory. Generated media remains
local/ignored; only this practical guide and source scripts belong in the
checkout.
