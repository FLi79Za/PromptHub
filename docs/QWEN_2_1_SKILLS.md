# Qwen 2.1 PromptHub skills

Two independent version 1.0.0 packages appear in Skills and Apply Skill:

- **Qwen 2.1 TXT2IMG** (`qwen-2-1-txt2img`): write a visual prompt from a brief; preserve counts, placement and exact lettering; return output settings separately.
- **Qwen 2.1 Image Edit** (`qwen-2-1-image-edit`): write a focused CHANGE / PRESERVE / MATCH instruction; identify canvas and ordered reference roles; preserve untargeted identity and geometry.

Choose your available Ollama writing model. The Qwen target is the model for which the prompt is written. Both packages support Create, Transform, Convert, Refine and Diagnose. No image renderer or model download is needed to write prompts.

Example TXT2IMG brief: A square botanical poster with exactly three red poppies and the heading "FIELD NOTES" at the top. Return the aspect ratio separately.

Example Image Edit brief: In <image1> (canvas), replace only the person's coat with the coat described for <image2> (garment reference). Preserve identity, pose, accessories and background. Match scene lighting and fabric shadows.

The current runtime accepts text inputs. Describe the source and reference roles for edit prompts; this does not add image upload, visual inspection or ComfyUI rendering. Paste the resulting prompt into your Qwen workflow. Transparency uses the documented RGBA wrapper. Local edits retain the canvas ratio; runtime settings remain outside the prompt.

Each package bundles the updated Codex Image Prompt Craft Qwen reference unchanged. provenance.json records its SHA-256 and the three knowledge record IDs. The explicit reference directive ensures it loads even when the brief and target omit the model name. Other existing PromptHub skills remain unchanged.

To install on another PromptHub instance, import each package directory through Skills or POST /api/skills/import with source set to an external copy of that directory. Do not import a directory over itself: the importer replaces its destination. Re-importing identical source is idempotent.

Validation: python -m unittest discover -s tests -p test_qwen_skills.py
The integration test uses a stub writer to verify import, catalogue discovery, all five operations, mandatory reference loading and completed traces. It does not establish image quality or run Qwen inference.
