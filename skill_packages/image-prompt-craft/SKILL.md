---
name: image-prompt-craft
description: Create, convert, refine and diagnose model-native prompts for text-to-image, reference-guided image generation, image-to-image transformation, inpainting, compositing and natural-language image editing. Use for Krea 2, Ideogram 4, FLUX.2 and FLUX.2 Klein, Z-Image, ChatGPT Images and GPT Image, Google Nano Banana and Imagen, Qwen-Image/Edit, or an unspecified image model. Also use to ingest official-source research digests or scout updates into this skill's specialised image-prompt knowledge base.
---

# Image Prompt Craft

Translate creative intent into the control language of the selected image model. Preserve the idea; change the syntax and emphasis to suit the model.

## Route the request

1. Determine the task mode: text-to-image, reference-guided generation, broad transformation, localised edit/inpaint, compositing, text replacement, or iterative continuation.
2. Use the named model. If unspecified, recommend one only when the user's priorities make the choice material; otherwise ask one concise question or provide a portable prompt plus model-specific variants.
3. Read [core-patterns.md](references/core-patterns.md).
4. Read only the matching model reference:
   - Krea 2: [krea-2.md](references/krea-2.md)
   - Ideogram 4: [ideogram-4.md](references/ideogram-4.md)
   - FLUX.2/Klein: [flux-2.md](references/flux-2.md)
   - Z-Image: [z-image.md](references/z-image.md)
   - ChatGPT Images/GPT Image: [gpt-image.md](references/gpt-image.md)
   - Nano Banana/Imagen: [google-images.md](references/google-images.md)
   - Qwen or local-model selection: [local-models.md](references/local-models.md)
5. Search [knowledge-updates.jsonl](references/knowledge-updates.jsonl) for the canonical model name and read relevant newer entries. Let a newer high-confidence official entry override an older reference rule. Flag unresolved contradictions.

## Produce the prompt

- Gather only missing constraints that materially affect the result: identity, action, framing, aspect ratio, scene, medium, lighting, exact visible text, reference roles and invariants.
- Use UK English unless the target model or required visible text needs another language.
- Return the ready-to-use prompt first.
- Add reference roles and edit invariants when images are supplied.
- Add settings notes only when officially documented and useful.
- Label community-derived 16 GB workarounds as experimental implementation guidance.
- Do not append generic negative prompts, quality-token soup, unsupported camera metadata, sampler settings or LoRA advice by habit.
- Do not turn a surgical edit into a cinematic rewrite. Keep the requested change narrow.

## Handle image edits

Express every edit through three decisions:

- `CHANGE`: the exact target, location and desired state.
- `PRESERVE`: identity, pose, geometry, framing, background, text or other invariants.
- `MATCH`: perspective, scale, colour temperature, grain, lighting, contact shadows and reflections.

Restate important invariants during iterative edits. Change one major variable per pass when identity, product geometry or layout matters.

## Import new research

Use this workflow when the user supplies a scout result, research digest, official documentation update or asks to update the knowledge base:

1. Read [import-policy.md](references/import-policy.md).
2. Resolve each claimed source. Prefer official documentation, official repositories and official model cards. Browse when the source is current, linked but unread, or not already present in the supplied material.
3. Extract only findings that change prompt structure, model routing, supported modes, reference handling, editing practice, text rendering, output controls or realistic 16 GB viability.
4. Normalise findings to the JSON schema in [update-schema.json](references/update-schema.json). Keep prompting rules separate from workflow/runtime notes.
5. Write a temporary JSON object or array and run `python3 scripts/import_updates.py path/to/updates.json`.
6. Inspect the result. Do not claim an import if validation fails. Re-run with `--dry-run` before importing uncertain batches.
7. When an update materially supersedes stable guidance, edit the appropriate model reference as well as recording the update. Preserve useful older advice unless the new official source invalidates it.
8. Validate and save the updated personal skill through the skill-creation workflow.

Do not import marketing-only announcements, duplicated advice, unattributed folklore, benchmark claims without prompting impact, or generic prompt recipes presented as model-specific facts.

## Output formats

For ordinary requests, return only sections that add value:

1. `Prompt`
2. `Reference roles` for supplied images
3. `Edit invariants` for edits/composites
4. `Settings note` when material
5. `Experimental note` for non-official implementation advice

For conversions, identify any source-model instruction that does not transfer cleanly, then provide the target-native prompt.
