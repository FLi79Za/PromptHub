# Qwen-Image-2.1 Prompt Craft

Use Qwen-Image-2.1 for local text-to-image, reference-guided generation and edits. Keep prompt craft separate from ComfyUI resolution, quantisation and offload settings.

## Text-to-image

- State the medium, main subject, background or palette, and every fixed object, count, colour and position. Use a complete visual description rather than quality-token lists.
- For a layout, poster, interface or infographic, describe the frame in reading order: background, top band, left/centre/right body, then lower band. For a single subject, describe background, placement and pose, then visible features, clothing or materials.
- Quote every string that must be readable exactly, preserving its characters, punctuation, spacing and script. State its location and typographic treatment. Do not invent legible text for decoration; call incidental text indistinct when it is not meant to be read.
- Select aspect ratio in the workflow output control, not in the image description. Qwen's official T2I rewriter emits the ratio as a separate field; use the user's requested ratio, otherwise choose the frame that suits the intended composition.
- The generator accepts direct prompts. When a brief is too short for a dense composition, use Qwen's optional T2I prompt rewriter to expand it into a detailed visual description; it is not required for ordinary prompts.

## Editing and image-to-image

Write a single direct instruction using this order:

`CHANGE: <target and desired state>. PRESERVE: <identity, untargeted content and composition>. MATCH: <only the lighting, perspective, shadows, grain or material behaviour needed for the altered region>.`

- Lead with the operation and make the named change strong and unambiguous. Preserve everything the user did not target; do not quietly clean up, regrade, reframe or redesign adjacent content.
- Preserve untouched content by role rather than redescribing it in detail. Over-describing an invariant can make the model regenerate and drift it. Treat a person's identity, distinctive accessories, product geometry and marks, and the input medium as high-priority invariants unless the edit targets them.
- For a local change, keep the source image's composition and aspect ratio. For a new scene that uses an input only as an identity reference, specify the new setting, lighting and composition, then choose an aspect ratio for that new scene instead.
- Use a circle, painted annotation or mask to identify a difficult local target, then state exactly what changes there. Prefer another local correction over widening an unsuccessful edit into a full-scene rewrite.

## References, compositing and text edits

- Give each supplied image one explicit role: `canvas` (the framing and untargeted content to retain), `identity source`, `garment/material source`, `style source`, or `background source`.
- In ComfyUI multi-image workflows, refer to ordered slots as `<image1>`, `<image2>`, and so on. Identify the canvas explicitly: it should normally be the target scene for compositing, the person for a clothing swap, the content image for style transfer, or the original image for a local replacement.
- For text replacement or addition, quote the exact rendered string and name the target language. If neither is supplied, retain the dominant language of readable text in the source image; if the source has no readable text, use the instruction language. Keep rendered text monolingual unless bilingual output is requested.
- Qwen officially supports up to ten reference images. ComfyUI exposes more ordered slots, but do not assume more than ten have the same model-level support without current implementation evidence.

## Transparency and practical output notes

- For a transparent asset, use Qwen's official wrapper exactly: `This is an RGBA image with transparency. <your description>. The image has alpha channel and the background is transparent.`
- In the official ComfyUI edit workflow, `custom_size` off follows the first image's aspect ratio. A custom canvas should remain close to the resized canvas image or the edit can shift. This is a workflow constraint, not text to add to the prompt.
- The official ComfyUI template uses an int8 diffusion model to reduce memory, but Qwen gives no 16 GB VRAM minimum. On the RTX 4070 Ti SUPER, treat Qwen-Image-2.1 as experimental and validate the specific resolution, references and offload configuration before relying on it.

## Official sources

- Qwen [T2I prompt-rewriter profile](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_t2i.txt)
- Qwen [editing prompt-rewriter profile](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_edit.txt)
- Qwen [model repository and prompt-rewriter setup](https://github.com/QwenLM/Qwen-Image-2.1)
- ComfyUI [native Qwen-Image-2.1 workflow](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1)
