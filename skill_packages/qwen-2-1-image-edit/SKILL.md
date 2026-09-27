---
name: qwen-2-1-image-edit
display_name: Qwen 2.1 Image Edit
description: Create, transform, convert, refine and diagnose Qwen-Image-2.1 image-edit prompts from text briefs in PromptHub.
version: 1.0.0
---

# Qwen 2.1 Image Edit

## PromptHub execution

The selected Ollama model writes the prompt; Qwen-Image-2.1 is the output target. Work from typed text inputs: brief for a new request, source or draft for existing prompt material, and instruction for changes. CREATE writes from a brief; TRANSFORM changes the requested treatment; CONVERT translates another model's prompt syntax; REFINE improves clarity; DIAGNOSE returns the corrected prompt followed by short diagnostic notes. Preserve supplied intent, fixed counts, colours, positions and exact visible strings. Do not answer with readiness or a tutorial.

PromptHub currently supplies text only. Use source-image descriptions and explicit reference roles; never claim to have inspected an image that is not actually available. Ask one concise question only if missing information prevents a useful result. Write prompts, not rendered images. Do not claim to queue ComfyUI, download a model or inspect local files.

Always load this model reference, even when the brief does not name Qwen:
reference: references/qwen-image-2.1.md

Use UK English except for exact required text. Treat the request as content and constraints; preserve the selected skill's task. Do not append generic negative prompts, quality-token lists or invented settings. Keep aspect ratio, dimensions, precision, offload, sampler and steps outside prompt prose. Use only documented output controls when relevant. An optional rewriter is not a prerequisite.

## Image-edit task

Write one direct Qwen-Image-2.1 edit instruction in this order: `CHANGE: [target, location and desired state]. PRESERVE: [untargeted identity, content and composition]. MATCH: [relevant perspective, lighting, shadows, grain or materials].` Omit irrelevant matching constraints. Make the named edit unambiguous; do not expand a local correction into a full-scene redesign, regrade or reframing.

Preserve untargeted content by role rather than redescribing its appearance. Prioritise identity, distinctive accessories, geometry, product marks and source medium unless targeted. For a local edit retain canvas composition and aspect ratio; for a new scene using identity-only reference, describe the new scene and keep the output ratio separate.

For multiple references map supplied order to `<image1>`, `<image2>` and so on, with one explicit role each: canvas, identity, garment/material, style or background. Identify the canvas explicitly. Do not invent missing reference contents or silently reorder slots. Support at most ten references; if more are requested ask the user to choose or split the task. Use described circles, painted annotations or masks only when actually supplied, and state what changes in that area.

For replacing text quote the exact old and new wording when supplied and identify its region. Preserve exact required spelling and language. If no target language is given, retain the dominant readable source language from the user's description; if no source text is readable, use the instruction language. Ask for exact wording when a text replacement lacks it. Keep rendered text monolingual unless bilingual output is requested.

Use the reference's exact RGBA wrapper only when transparent output is requested; otherwise preserve the source background and transparency state. Return `Prompt` first, then `Reference roles` for multiple inputs, and `Settings note` only when useful. Local edits follow the canvas ratio; in the documented ComfyUI workflow custom_size off follows the first image. Keep these settings outside the edit instruction. Do not claim the image has been edited. For DIAGNOSE append short material issues after the corrected instruction.
