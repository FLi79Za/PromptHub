# Google Nano Banana and Imagen

## Current model status

As of 26 August 2026, Google documents Imagen as deprecated and shut down on 17 August 2026. Do not route new work to Imagen or present its 480-token prompt guide as an active model option. Route current Google image generation to Nano Banana variants instead.

## Current Nano Banana routing

Google now uses Nano Banana for four distinct models:

- **Nano Banana 2** (`gemini-3.1-flash-image`): default general-purpose route. Prefer it for ordinary generation, multi-reference work, character consistency, reliable text, 4K output, Google Image Search grounding and video-to-image.
- **Nano Banana 2 Lite** (`gemini-3.1-flash-lite-image`): fastest and cheapest route. Prefer it for simple, high-volume generation or editing, but not for multiple reference inputs or long sequential edits. It supports 1K output only.
- **Nano Banana Pro** (`gemini-3-pro-image`): use for complex professional assets, brand consistency, localisation, advanced grounding and precision creative control.
- **Nano Banana** (`gemini-2.5-flash-image`): legacy route. Google recommends migrating to Nano Banana 2 Lite rather than choosing it for new work.

## Shared prompting guidance

- Treat Gemini-native image generation as conversational. Iterate in context instead of rebuilding the full prompt for every turn.
- For generation, specify subject, composition, action, location and style. Add camera angle, lighting and visible wording when relevant.
- For a surgical edit, give a short direct change followed by explicit invariants.
- Assign roles to reference images: character, object, scene, pose or style.
- For text-heavy work, finalise or generate the wording first, then request the image containing that exact text.
- Choose the model by reference role and workflow complexity, not by the total input count alone.
- For iterative localisation or redesign, request only the delta and say not to change other elements.
- Use exposed aspect-ratio and output-size controls rather than burying them only in prose.

## Reference and output capabilities

- Gemini 3 image models accept up to 14 total reference images, but their role-specific capacities differ.
- Nano Banana 2 supports high-fidelity use of up to 10 object references and resemblance for up to four characters.
- Nano Banana Pro supports up to six object references, five character references and three style references.
- Nano Banana 2 Lite accepts up to 14 object images, but does not provide the character- or style-reference routes documented for the other Gemini 3 image models.
- Assign every input a role such as character, object, scene, style or video, and state what to preserve or transfer from it.
- Nano Banana 2 alone supports Google Image Search grounding and video-to-image. Use those capabilities for current factual visual grounding or for a still derived from video context.
- Gemini 3 image models support 1K, 2K and 4K output; Nano Banana 2 also supports 512 px, while Lite supports 1K only. Set aspect ratio and image size through output controls.

Official sources:

- https://ai.google.dev/gemini-api/docs/image-generation
- https://blog.google/products-and-platforms/products/gemini/prompting-tips-nano-banana-pro/

## Imagen

- Legacy/deprecated: do not recommend for new work. Migrate to Nano Banana.
- Structure prompts as `subject + context + style`.
- Use clear descriptive keywords and modifiers rather than story-like prose.
- Keep within the documented 480-token prompt limit.
- Keep Imagen distinct from conversational Nano Banana editing.

Official source: https://ai.google.dev/gemini-api/docs/imagen
