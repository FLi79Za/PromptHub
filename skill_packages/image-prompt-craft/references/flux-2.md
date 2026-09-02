# FLUX.2 and FLUX.2 Klein

- Build prompts around `subject + action + style + context`, then add composition, lighting and colour where they affect the image.
- Give Klein complete descriptive prompts because it does not include prompt upsampling.
- Use positive descriptions instead of negative prompts. Describe the desired replacement state.
- Use exact colour language or hex codes for brand work when the interface supports colour steering.
- For edits, state the target change and explicitly preserve camera angle, framing, identity, unaffected objects and background.
- For composites, assign each reference a role and require perspective, scale, illumination, contact shadows and reflections to match the base.
- Avoid conflicting aesthetics and camera-spec lists that do not change the visual intention.

## Text-to-image pattern

```text
[specific subject] [action/pose], [style or medium], in [context]. [composition and viewpoint]. [lighting and palette]. [material and texture details]. [positive constraint state].
```

## Edit pattern

```text
Change [target and location] to [desired state]. Preserve [identity, geometry, framing and unaffected scene]. Match the original [perspective, lighting, shadows, reflections and texture].
```

## Local note

Track 4B and 9B Klein variants for 16 GB workflows. Black Forest Labs documents consumer-hardware operation with as little as 13 GB VRAM for Klein, making it a strong 16 GB candidate. Exact memory and speed still depend on precision, offload, resolution and frontend. Keep these runtime factors outside the generated prompt.

Official sources:

- https://docs.bfl.ai/guides/prompting_summary
- https://docs.bfl.ai/guides/prompting_guide_flux2
- https://docs.bfl.ai/guides/prompting_guide_t2i_negative
- https://docs.bfl.ai/flux_2/flux2_overview
- https://docs.bfl.ai/release-notes
