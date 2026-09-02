# ChatGPT Images and GPT Image

## Current model status

- Use `gpt-image-2` as the current OpenAI API target for new image generation and editing workflows.
- OpenAI's 2026 deprecation schedule routes `gpt-image-1`, `gpt-image-1-mini`, `gpt-image-1.5` and `chatgpt-image-latest` users to `gpt-image-2`. Do not recommend those older aliases for new API builds.
- In the Responses API, the mainline model invokes the image-generation tool and model selection is handled by the tool. Use the Image API when the user needs a single direct generation or edit with explicit GPT Image model selection.

- Call the system ChatGPT Images or GPT Image, not Imagen.
- Use clear intent, explicit constraints and a stable skimmable structure. Paragraph, tag and JSON-like formats can all work.
- For edits, state the change and restate invariants on every turn. Prefer small iterative changes.
- For multi-image composition, identify each source by role, state what to transplant and where, and preserve the untouched scene.
- Require inserted elements to match perspective, scale, lighting, shadows and reflections.
- Quote exact wording, request one occurrence, and define hierarchy, typography, contrast and placement.
- Describe natural skin texture, believable materials and plausible lighting instead of relying on `photorealistic` alone.
- In the ChatGPT editor, name the affected area even when using a selection because selection boundaries may not be exact.
- For transparent assets, request an isolated subject on a fully transparent background and preserve transparency in later edits.

## GPT Image 2 controls

- GPT Image 2 processes all image inputs at high fidelity automatically. Do not advise setting `input_fidelity`; the API does not accept it for this model.
- Continue to identify multiple inputs by role and state what to transplant, preserve and match. Automatic high fidelity does not replace prompt-level role assignment.
- Use the API's flexible `size` control for exact output dimensions and `quality` for the fidelity-latency trade-off. Keep these settings outside the visual prompt.
- Transparent backgrounds are in preview. Set `background` to `transparent`, use PNG or WebP, explicitly request a fully transparent background, and preserve it in every subsequent edit.
- Masks guide the edit but are not exact pixel boundaries. Name the intended target area and restate unaffected regions even when a mask is supplied.

Official sources:

- https://developers.openai.com/cookbook/examples/multimodal/image-gen-models-prompting-guide
- https://developers.openai.com/api/docs/guides/image-generation
- https://developers.openai.com/api/docs/deprecations
- https://help.openai.com/en/articles/11084440-images-in-chatgpt
