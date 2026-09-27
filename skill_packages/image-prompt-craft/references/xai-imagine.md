# Grok Imagine

## Current routing

- Use `grok-imagine-image-2.0` for new text-to-image and image-editing work.
- Do not select `grok-imagine-image-quality` for new work. xAI retires that slug on 2 November 2026 and then redirects it to `grok-imagine-image-2.0` at `quality: "low"`. Migrate explicitly so quality remains intentional.
- `quality: "auto"` currently resolves to `low` for generation and `medium` for editing. Set `low` or `medium` when cost or fidelity must be predictable.

## Prompt and edit structure

- Use natural-language scene descriptions for generation. Put aspect ratio, resolution and quality in their exposed controls instead of relying on prompt wording alone.
- For editing, use the shared `CHANGE / PRESERVE / MATCH` structure. xAI supports iterative editing by passing each result into the next request.
- Up to five source images can be combined in one edit. Keep their request order deliberate and identify the intended subject or contribution of each source in the prompt when ambiguity is possible.
- By default, an edit follows the first source image's aspect ratio. Override it explicitly when the desired composition differs.

## Output controls

- Generation can return 1–10 images per prompt.
- Supported resolutions are 1K and 2K.
- In addition to common portrait and landscape ratios, `grok-imagine-image-2.0` supports `21:9` and `5:2` wide output.

## Hardware note

This is a hosted xAI API model. Do not treat it as a local 16 GB VRAM option.

Official sources:

- https://docs.x.ai/developers/migration/imagine-image-quality-nov-2
- https://docs.x.ai/developers/model-capabilities/images/generation
- https://docs.x.ai/developers/model-capabilities/images/editing
- https://docs.x.ai/developers/model-capabilities/images/multi-image-editing
